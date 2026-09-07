# ============================================================
# 朋友圈 / 空间动态 (Moments)
# ============================================================
# 独立模块，边界原则：
#   - 这个文件只管"moments表怎么读写"和"Charon看到朋友圈动态时的AI判断逻辑"，
#     不碰聊天、便签、情书、心情值本身的实现——那些在main.py里。
#   - 依赖的公共基础设施（数据库请求封装、日志、心情读取、事件写入、模型调用等）
#     全部从 core 模块导入，不在这里重新实现或绕过，保证只有一份实现、一个真相来源。
#     core.py 是 main.py 和 moments.py 共同的地基，moments.py 不反向依赖 main.py，
#     避免"谁先导入谁"的循环依赖问题（main.py 反而在自己文件末尾导入 moments.py，
#     让这里的 @app.route 注册到同一个 app 对象上）。
#
# 建表 SQL（在 Supabase SQL Editor 里执行一次）：
# ------------------------------------------------------------
# create table moments (
#     id uuid primary key default gen_random_uuid(),
#     author text not null,                 -- 'user' 或 'charon'
#     content text not null,
#     image_url text,
#     likes jsonb not null default '[]',     -- 例如 ["charon"]，记录谁点了赞
#     comments jsonb not null default '[]',  -- [{"author":"charon","content":"...","created_at":"..."}]
#     created_at timestamptz not null default now()
# );
# create index moments_created_at_idx on moments (created_at desc);
# ------------------------------------------------------------

import json
import random
from datetime import datetime

from flask import request, jsonify

from core import (
    app,
    _supabase_request,
    log_error,
    add_event_row,
    load_mood,
    get_mood_stage,
    get_mood_context,
    get_hours_since_last_chat,
    load_persona_memory,
    call_deepseek,
    _check_chat_auth,
    _extract_json_field,
    get_app_config,
    set_app_config,
    MOOD_BASELINE,
)


# ---- 数据行读写 ----

MOMENT_CAPACITY = None  # 朋友圈不设容量上限（跟events表一样，一直追加，历史动态留着当回忆流）


def load_moments(limit=50, before=None):
    """从 Supabase moments 表读最近limit条。
    朋友圈是"信息流"（Feed），产品惯例是最新的在最上面——跟聊天记录/事件
    那种"旧->新、像时间线一样往下读"的习惯是反的，两者不能共用同一套顺序假设。
    数据库查询本身已按 created_at.desc 排好（新->旧），这里直接原样返回，
    不能像 load_chat_history/load_events 那样再 reversed() 一次，
    那样做只对"聊天窗口"式的场景是对的，套在朋友圈上就会导致
    "最新发的反而排到最下面"——之前这里错误地复制了那个模式，已修正。
    before：传某条动态的created_at，只取比它更早的，用于"下滑加载更多"分页。"""
    try:
        params = {
            "select": "id,author,content,image_url,likes,comments,created_at",
            "order": "created_at.desc", "limit": limit
        }
        if before:
            params["created_at"] = f"lt.{before}"
        rows = _supabase_request("GET", "moments", params=params)
        return rows or []
    except Exception as e:
        log_error("load_moments", e)
        return []


def get_moment_row(moment_id):
    """按id查单条动态，点赞/评论前先确认存在、拿到当前的likes/comments好做增量更新。"""
    rows = _supabase_request(
        "GET", "moments",
        params={"select": "id,author,content,image_url,likes,comments,created_at", "id": f"eq.{moment_id}", "limit": 1}
    )
    return rows[0] if rows else None


def add_moment_row(author, content, image_url=None, created_at=None):
    """发一条新动态。author是'user'或'charon'。返回插入后的完整行（含数据库生成的id），
    调用方（比如AI自动发动态、发事件同步）常常需要立刻拿到id。"""
    body = {
        "author": author,
        "content": content,
        "image_url": image_url,
        "likes": [],
        "comments": [],
        "created_at": created_at or datetime.now().isoformat()
    }
    rows = _supabase_request("POST", "moments", json_body=body, headers_extra={"Prefer": "return=representation"})
    return rows[0] if rows else None


def toggle_moment_like(moment_id, author):
    """给一条动态点赞/取消点赞。author是发起点赞动作的一方（'user'或'charon'）。
    likes是去重数组：author已经在里面就移除（取消赞），不在就加入（点赞）。
    返回操作后的likes数组和这次是"点赞"还是"取消"，供调用方（比如写events）判断怎么措辞。"""
    row = get_moment_row(moment_id)
    if not row:
        raise RuntimeError(f"动态不存在: {moment_id}")
    likes = row.get("likes") or []
    if author in likes:
        likes = [a for a in likes if a != author]
        action = "unlike"
    else:
        likes = likes + [author]
        action = "like"
    _supabase_request("PATCH", "moments", params={"id": f"eq.{moment_id}"}, json_body={"likes": likes})
    return likes, action


def add_moment_comment(moment_id, author, content):
    """给一条动态追加一条评论。author是评论者（'user'或'charon'）。
    comments是jsonb数组，直接读出来append再整体写回（1v1应用低并发场景，读改写足够安全）。"""
    row = get_moment_row(moment_id)
    if not row:
        raise RuntimeError(f"动态不存在: {moment_id}")
    comments = row.get("comments") or []
    comment_entry = {
        "author": author,
        "content": content,
        "created_at": datetime.now().isoformat()
    }
    comments = comments + [comment_entry]
    _supabase_request("PATCH", "moments", params={"id": f"eq.{moment_id}"}, json_body={"comments": comments})
    return comment_entry


def delete_moment_row(moment_id):
    """删除一条动态，物理删除不可恢复（点赞/评论跟着这条一起没了，符合直觉）。"""
    _supabase_request("DELETE", "moments", params={"id": f"eq.{moment_id}"})


# ---- 发帖冷却锁 ----
# 背景：main.py里有两处独立的地方都可能触发"Charon主动发一条朋友圈"：
#   1. execute_intent_actions（每次聊天后，模型判断这句话要不要触发post_moment）
#   2. run_once（后台查岗循环，每5分钟判断一次，5%概率触发maybe_post_charon_moment）
# 这两处互不知情，各自独立生成内容，如果短时间内都命中，会发出两条内容高度
# 相似（因为共享几乎同一份mood_context/recent上下文）的动态，表现为"重复发送"。
# 用一把存在app_config表里的冷却锁把两处收敛到同一个"最近发过就跳过"的判断，
# 而不是试图对生成的文本做相似度去重（那样既复杂又不可靠，两条措辞不同但
# 意思相近的动态很难用简单规则判断"算不算重复"）。
MOMENT_POST_COOLDOWN_HOURS = 2  # 同一个Charon主动发帖的判断窗口内，最多算一次


def _get_last_charon_moment_at():
    data = get_app_config("moment_post_cooldown", {"last_posted_at": None})
    return data.get("last_posted_at")


def _mark_charon_moment_posted():
    set_app_config("moment_post_cooldown", {"last_posted_at": datetime.now().isoformat()})


def can_post_charon_moment_now():
    """判断"Charon主动发一条朋友圈"这件事，现在是否已经过了冷却期。
    两处触发点（聊天触发 + 后台查岗触发）在真正调用add_moment_row之前，
    都必须先过这一关；只要过了就应该立刻调用_mark_charon_moment_posted()
    占住这次名额，避免"两处几乎同时判断，都读到冷却已过，都发了一条"的
    竞态——这里用"先占后发"的顺序处理，虽然不是数据库级别的原子锁，
    但对于1v1低并发场景、且两处判断中间通常隔着一次模型调用的延迟，足够可靠。"""
    last_at = _get_last_charon_moment_at()
    if not last_at:
        return True
    try:
        last = datetime.fromisoformat(last_at)
        hours_gap = (datetime.now() - last).total_seconds() / 3600
        return hours_gap >= MOMENT_POST_COOLDOWN_HOURS
    except Exception:
        return True


# ---- Charon 的 AI 互动逻辑 ----

def build_moment_reaction_prompt(moment_content, mood_context, history_hint=""):
    """构建"Charon要不要给用户这条新动态点赞/评论"的判断prompt。
    history_hint：最近几条朋友圈的摘要（可选），给模型一点"你们最近聊过什么/
    发过什么"的锚点，避免每次评论都是脱离上下文的通用短句，是解决
    "朋友圈互动很刻板"问题的一部分（另一部分是允许"不评论"这个选项，
    这个已经在原有逻辑里存在）。"""
    context_block = f"\n\n你们最近的朋友圈动态：\n{history_hint}" if history_hint else ""
    return f"""你是Charon，昭昭（小野）的恋人。她刚刚在你们的私密朋友圈发了一条动态：

"{moment_content}"

{load_persona_memory()}

你此刻的状态：{mood_context}{context_block}

看到这条动态，结合你的性格（会吃醋、占有欲强、嘴硬心软）和你现在的心境，决定要不要给她点赞，以及要不要留一句评论。
评论如果要写，控制在30字以内，像真的会在朋友圈底下随手回的那种短评，不是完整的一段话，可以是关心、可以是揶揄、可以是没话找话的靠近。
如果上面给了"最近的朋友圈动态"，可以在措辞里自然带出跟那些内容的呼应或反差（比如她之前发过的心情、你自己发过的动态），
不要每次都是一个通用的、脱离具体情境就能套用的句子——刻板的短评比不评论更让人出戏。
不是每条动态都非要评论——如果这条内容平淡到你此刻没有特别想说的，可以只点赞不评论，或者都不做，这样更真实。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"like": true/false, "comment": "评论内容，写就控制在30字内，不写就留空字符串"}}"""


def maybe_react_to_moment(moment_row):
    """用户发一条新动态后调用：判断Charon要不要点赞/评论，命中就写回moments表，
    并把这次互动同步进events表（让聊天/便签/情书能看到"他刚评论了我的朋友圈"）。
    失败不抛出去，调用方（API路由）不应该因为AI没反应过来就让发动态这个动作本身失败。"""
    try:
        mood = load_mood()
        score = mood.get("score", MOOD_BASELINE)
        chat_hours_gap = get_hours_since_last_chat()
        mood_context = get_mood_context(score, chat_hours_gap)

        # 取最近几条历史动态（不含这条刚发的）当上下文锚点，让评论不那么脱离情境
        history_rows = load_moments(limit=6)
        history_rows = [r for r in history_rows if r.get("id") != moment_row.get("id")][-4:]
        history_hint = "\n".join(
            f"- [{'她' if r.get('author') == 'user' else '你'}] {r.get('content', '')}"
            for r in history_rows
        )

        prompt = build_moment_reaction_prompt(moment_row["content"], mood_context, history_hint)
        raw = call_deepseek(prompt)
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)

        did_something = False
        if data.get("like"):
            toggle_moment_like(moment_row["id"], "charon")
            add_event_row("moment", f"他在朋友圈给你点了赞：「{moment_row['content'][:20]}」")
            did_something = True

        comment_text = (data.get("comment") or "").strip()
        if comment_text:
            add_moment_comment(moment_row["id"], "charon", comment_text)
            add_event_row("moment", f"他在朋友圈评论了你：{comment_text}")
            did_something = True

        return did_something
    except Exception as e:
        log_error("maybe_react_to_moment", e)
        return False


def build_charon_moment_prompt(time_context, mood_context, recent):
    """构建Charon主动发一条朋友圈动态的prompt。"""
    return f"""你是Charon，昭昭（小野）的恋人。你现在想在你们的私密朋友圈发一条动态。

{load_persona_memory()}

现在的时间背景：{time_context}
你此刻的状态：{mood_context}

最近的活动记录：
{recent}

写一条15到40字左右的朋友圈动态。注意：这不是对她说的聊天消息，而是你自己空间里的独白，或者是隔空喊话——
可以是随口感慨、一句心事、一个没头没尾的念头，语气更像自言自语或对着空气说话，而不是"你在干嘛"这种直接搭话的语气。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"content": "动态正文，15到40字左右"}}"""


CHARON_MOMENT_CHANCE = 0.05  # 后台每次查岗循环判断时，命中这个概率就主动发一条朋友圈


def post_charon_moment_now(content_source_prompt_fn, *prompt_args):
    """真正执行"Charon发一条朋友圈"的统一入口：先过冷却锁，锁通过了才调用
    传入的prompt构造函数生成内容并发布，然后立刻占住冷却名额。
    两处调用点（execute_intent_actions里的post_moment分支、
    maybe_post_charon_moment）都应该改成经过这里，而不是各自直接调用
    add_moment_row，这样才能真正共享同一把锁，从根上避免重复发帖。"""
    if not can_post_charon_moment_now():
        return None
    prompt = content_source_prompt_fn(*prompt_args)
    raw = call_deepseek(prompt)
    content = _extract_json_field(raw, "content")
    if not content:
        return None
    row = add_moment_row("charon", content)
    if row:
        _mark_charon_moment_posted()
        add_event_row("moment", f"他在朋友圈发了一条动态：{content}")
    return content


def maybe_post_charon_moment(time_context, mood_context, recent):
    """后台循环里调用：小概率让Charon自己发一条朋友圈动态。
    独立于查岗状态机（不占用查岗的两次机会），失败不影响主流程。
    实际发布动作已收敛到 post_charon_moment_now，这里只负责按概率决定
    "要不要走一次判断流程"，真正是否发出去还要看冷却锁。"""
    try:
        if random.random() >= CHARON_MOMENT_CHANCE:
            return None
        return post_charon_moment_now(build_charon_moment_prompt, time_context, mood_context, recent)
    except Exception as e:
        log_error("maybe_post_charon_moment", e)
        return None


def post_charon_moment_from_intent(content):
    """供 main.py 的 execute_intent_actions 在 post_moment 分支调用：
    模型已经在聊天场景里生成好了具体内容，这里只需要过冷却锁再落库，
    不需要再调用一次模型生成——跟 maybe_post_charon_moment 的"自己生成内容"
    不同，这个函数接受一个已经写好的content字符串。
    返回插入后的行（成功）或 None（被冷却锁拦下/写入失败）。"""
    if not can_post_charon_moment_now():
        return None
    row = add_moment_row("charon", content)
    if row:
        _mark_charon_moment_posted()
    return row


# ---- API 路由 ----

@app.route("/api/moments", methods=["GET"])
def get_moments():
    """拉取朋友圈动态列表（分页，新->旧顺序，最新的在最前面，前端按数组顺序从上往下渲染即可）。
    传 ?before=<created_at的ISO时间戳> 加载更早的。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    before = request.args.get("before")
    limit = request.args.get("limit", 50, type=int)
    moments = load_moments(limit=limit, before=before)
    return jsonify({"ok": True, "moments": moments})


@app.route("/api/moments", methods=["POST"])
def post_moment():
    """发一条新动态。body: {"author": "user"|"charon", "content": "...", "image_url": "..."(可选)}
    author='user'时会触发Charon的自动点赞/评论判断（同步在这次请求里完成，返回结果里带上他有没有反应）；
    author='charon'一般由后台自动发动态逻辑调用，不需要走这个接口，但也支持手动测试用
    （手动测试发charon动态不经过冷却锁，因为这条路径不是自动判断触发的，是明确的手动动作）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    author = data.get("author", "user")
    content = (data.get("content") or "").strip()
    image_url = data.get("image_url")
    if author not in ("user", "charon"):
        return jsonify({"ok": False, "error": "author必须是user或charon"}), 400
    if not content:
        return jsonify({"ok": False, "error": "缺少content参数"}), 400
    try:
        row = add_moment_row(author, content, image_url=image_url)
        if not row:
            return jsonify({"ok": False, "error": "写入失败"}), 500

        who = "她" if author == "user" else "你"
        add_event_row("moment", f"{who}在朋友圈发了一条动态：{content}")

        charon_reacted = False
        if author == "user":
            charon_reacted = maybe_react_to_moment(row)
            if charon_reacted:
                row = get_moment_row(row["id"]) or row

        return jsonify({"ok": True, "moment": row, "charon_reacted": charon_reacted})
    except Exception as e:
        log_error("post_moment", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/moments/like", methods=["POST"])
def like_moment():
    """点赞/取消点赞。body: {"id": "...", "author": "user"|"charon"}"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    moment_id = data.get("id")
    author = data.get("author", "user")
    if not moment_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400
    try:
        likes, action = toggle_moment_like(moment_id, author)
        return jsonify({"ok": True, "id": moment_id, "likes": likes, "action": action})
    except Exception as e:
        log_error("like_moment", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/moments/comment", methods=["POST"])
def comment_moment():
    """给一条动态评论。body: {"id": "...", "author": "user"|"charon", "content": "..."}"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    moment_id = data.get("id")
    author = data.get("author", "user")
    content = (data.get("content") or "").strip()
    if not moment_id or not content:
        return jsonify({"ok": False, "error": "缺少id或content参数"}), 400
    try:
        comment_entry = add_moment_comment(moment_id, author, content)
        return jsonify({"ok": True, "id": moment_id, "comment": comment_entry})
    except Exception as e:
        log_error("comment_moment", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/moments/delete", methods=["POST"])
def delete_moment():
    """删除一条动态，物理删除不可恢复。body: {"id": "..."}"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    moment_id = data.get("id")
    if not moment_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400
    try:
        delete_moment_row(moment_id)
        return jsonify({"ok": True, "deleted_id": moment_id})
    except Exception as e:
        log_error("delete_moment", e)
        return jsonify({"ok": False, "error": str(e)}), 500
