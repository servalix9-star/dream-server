# ============================================================
# 账号主页 (Profile)：头像 / 昵称 / 聊天背景
# ============================================================
# 独立模块。user_profile表只有两行（'user'和'charon'），这里提供读取
# 和更新的接口。主要由"我"这边操作（改自己的资料，也代Charon改TA的资料），
# 但接口本身不区分"谁在操作"，只看body里传的user_id改哪一行——
# 这是个人使用的双人应用，不需要做"只有本人能改自己资料"这种权限隔离。
#
# 依赖core.py的公共基础设施（包括图片上传封装），不反向依赖main.py。
#
# 建表SQL见 sql/schema_additions.sql 里的 user_profile 部分。

from datetime import datetime

from flask import request, jsonify

from core import (
    app,
    _supabase_request,
    log_error,
    _check_chat_auth,
    upload_image_to_storage,
    delete_image_from_storage,
)


VALID_USERS = ("user", "charon")


def load_profile(user_id):
    rows = _supabase_request(
        "GET", "user_profile",
        params={"select": "user_id,nickname,avatar_url,chat_background,updated_at", "user_id": f"eq.{user_id}", "limit": 1}
    )
    return rows[0] if rows else None


def load_all_profiles():
    """一次拿双方资料，聊天页面初始化时用（要同时知道用户和Charon的头像/昵称）。
    返回 {"user": {...}, "charon": {...}} 结构，前端不用关心底层是列表还是字典。"""
    rows = _supabase_request(
        "GET", "user_profile",
        params={"select": "user_id,nickname,avatar_url,chat_background,updated_at"}
    ) or []
    return {r["user_id"]: r for r in rows}


def update_profile(user_id, nickname=None, avatar_url=None, chat_background=None):
    """局部更新：只传了哪个字段就只更新哪个字段，没传的字段保持原样。
    这跟moments/wallet那种"整条记录一次性写入"不同，是因为改头像、改昵称、
    改背景是用户体感上三个独立的动作（点头像才弹头像选择，点昵称才弹改名框），
    没必要要求调用方每次都把其他没变的字段也传一遍。"""
    if user_id not in VALID_USERS:
        raise ValueError("user_id 必须是 user 或 charon")

    body = {"user_id": user_id, "updated_at": datetime.now().isoformat()}
    if nickname is not None:
        body["nickname"] = nickname
    if avatar_url is not None:
        body["avatar_url"] = avatar_url
    if chat_background is not None:
        body["chat_background"] = chat_background

    # upsert：user_profile表在初始化SQL里已经预先插入了两行，正常情况下
    # 这里走的都是"更新已存在的行"，但用merge-duplicates兜底，即便某种原因
    # 那两行初始数据没插入成功，第一次调用这个接口也能自愈成正确状态。
    _supabase_request(
        "POST", "user_profile",
        json_body=body,
        headers_extra={"Prefer": "resolution=merge-duplicates"}
    )
    return load_profile(user_id)


# ---- API 路由 ----

@app.route("/api/profile", methods=["GET"])
def get_profile():
    """拉取资料。不传?user_id则返回双方资料的字典；传了就只返回那一方。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    user_id = request.args.get("user_id")
    if user_id:
        if user_id not in VALID_USERS:
            return jsonify({"ok": False, "error": "user_id必须是user或charon"}), 400
        return jsonify({"ok": True, "profile": load_profile(user_id)})
    return jsonify({"ok": True, "profiles": load_all_profiles()})


@app.route("/api/profile/update", methods=["POST"])
def post_profile_update():
    """更新昵称/聊天背景（纯文本/URL字符串字段，不涉及文件上传走这个接口）。
    body: {"user_id": "user"|"charon", "nickname": "..."(可选), "chat_background": "..."(可选)}
    换头像走单独的 /api/profile/avatar 接口（因为那个是文件上传，multipart表单）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    user_id = data.get("user_id")
    nickname = data.get("nickname")
    chat_background = data.get("chat_background")
    if nickname is None and chat_background is None:
        return jsonify({"ok": False, "error": "至少需要传nickname或chat_background之一"}), 400
    try:
        profile = update_profile(user_id, nickname=nickname, chat_background=chat_background)
        return jsonify({"ok": True, "profile": profile})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        log_error("post_profile_update", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/profile/avatar", methods=["POST"])
def post_profile_avatar():
    """换头像。multipart/form-data，字段：file（图片文件，必填）、user_id（'user'或'charon'）。
    上传新头像成功后，顺手清理旧头像文件（如果旧头像也是Storage里的图片，
    不是初始的/static/默认头像——delete_image_from_storage内部会自己判断跳过非Storage的URL）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    file = request.files.get("file")
    user_id = request.form.get("user_id")
    if not file:
        return jsonify({"ok": False, "error": "缺少file文件"}), 400
    if user_id not in VALID_USERS:
        return jsonify({"ok": False, "error": "user_id必须是user或charon"}), 400
    try:
        old_profile = load_profile(user_id)
        old_avatar = old_profile.get("avatar_url") if old_profile else None

        new_avatar_url = upload_image_to_storage(file.read(), file.filename, folder="avatars")
        profile = update_profile(user_id, avatar_url=new_avatar_url)

        if old_avatar and old_avatar != new_avatar_url:
            delete_image_from_storage(old_avatar)

        return jsonify({"ok": True, "profile": profile})
    except Exception as e:
        log_error("post_profile_avatar", e)
        return jsonify({"ok": False, "error": str(e)}), 500
