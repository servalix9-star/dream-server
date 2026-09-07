# ============================================================
# 心意币 / 转账 (Wallet)
# ============================================================
# 独立模块，边界原则同moments.py：只管"wallet_transfers表怎么读写"和
# "余额怎么算"，不碰聊天消息本身怎么落库（那是main.py的chat_send负责，
# 这里只提供"执行一笔转账"这个动作，返回结果交给调用方决定要不要
# 顺手写一条chat_messages记录）。
#
# 依赖core.py的公共基础设施，不反向依赖main.py，避免循环导入
# （跟moments.py保持一致的组织方式，main.py在文件末尾import wallet）。
#
# 建表SQL见 sql/schema_additions.sql 里的 wallet_transfers 部分。

from datetime import datetime

from flask import request, jsonify

from core import (
    app,
    _supabase_request,
    log_error,
    add_event_row,
    _check_chat_auth,
)


VALID_USERS = ("user", "charon")


def get_balance(who):
    """算某一方当前余额：所有以TA为收款方的金额之和，减去所有以TA为付款方的金额之和。
    不单独存"当前余额"字段，永远从流水表现算——保证余额和流水历史不可能出现不一致
    （比如某次写余额字段失败了，但流水记录成功了，就会出现对不上的情况，这里从设计上排除这种可能）。"""
    try:
        incoming = _supabase_request(
            "GET", "wallet_transfers",
            params={"select": "amount", "to_user": f"eq.{who}"}
        ) or []
        outgoing = _supabase_request(
            "GET", "wallet_transfers",
            params={"select": "amount", "from_user": f"eq.{who}"}
        ) or []
        return sum(r["amount"] for r in incoming) - sum(r["amount"] for r in outgoing)
    except Exception as e:
        log_error(f"get_balance:{who}", e)
        return 0


def load_transfers(limit=50, before=None):
    """读最近limit笔转账记录，旧->新顺序（跟聊天记录一致的时间线阅读习惯，
    这是"流水"不是"信息流"，展示逻辑更接近chat_messages而不是moments）。"""
    try:
        params = {
            "select": "id,from_user,to_user,amount,message,source,created_at",
            "order": "created_at.desc", "limit": limit
        }
        if before:
            params["created_at"] = f"lt.{before}"
        rows = _supabase_request("GET", "wallet_transfers", params=params)
        return list(reversed(rows or []))
    except Exception as e:
        log_error("load_transfers", e)
        return []


def create_transfer(from_user, to_user, amount, message=None, source="manual"):
    """执行一笔转账，插入流水记录。不做"余额不够就拒绝"的限制——见schema注释，
    这不是要严格模拟真实钱包的资金约束，是记录一份心意留痕，允许"透支"，
    余额为负也只是数字上的负，不代表任何实际意义上的欠款。
    amount必须是正整数（数据库check约束兜底），from/to必须是user或charon。"""
    if from_user not in VALID_USERS or to_user not in VALID_USERS:
        raise ValueError("from_user/to_user 必须是 user 或 charon")
    if from_user == to_user:
        raise ValueError("不能给自己转账")
    if not isinstance(amount, int) or amount <= 0:
        raise ValueError("amount 必须是正整数")

    body = {
        "from_user": from_user,
        "to_user": to_user,
        "amount": amount,
        "message": (message or "").strip() or None,
        "source": source,
        "created_at": datetime.now().isoformat(),
    }
    rows = _supabase_request("POST", "wallet_transfers", json_body=body, headers_extra={"Prefer": "return=representation"})
    row = rows[0] if rows else None

    if row:
        who = "她" if from_user == "user" else "你"
        note = f"（附言：{message}）" if message else ""
        add_event_row("wallet", f"{who}转了{amount}心意币给{'你' if to_user == 'charon' else '她'}{note}")

    return row


# ---- API 路由 ----

@app.route("/api/wallet/balance", methods=["GET"])
def get_wallet_balance():
    """拿双方当前余额，前端聊天页/账号主页展示钱包用。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return jsonify({
        "ok": True,
        "balance": {
            "user": get_balance("user"),
            "charon": get_balance("charon"),
        }
    })


@app.route("/api/wallet/transfers", methods=["GET"])
def get_wallet_transfers():
    """拉取转账流水（分页，旧->新顺序）。传 ?before=<created_at的ISO时间戳> 加载更早的。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    before = request.args.get("before")
    limit = request.args.get("limit", 50, type=int)
    transfers = load_transfers(limit=limit, before=before)
    return jsonify({"ok": True, "transfers": transfers})


@app.route("/api/wallet/transfer", methods=["POST"])
def post_wallet_transfer():
    """发起一笔转账。body: {"from": "user"|"charon", "to": "user"|"charon",
    "amount": 520, "message": "..."(可选)}
    这个接口只负责钱包本身的转账动作；如果前端想让这笔转账同时在聊天窗口里
    显示成一条消息气泡，需要调用方（前端）自己再调一次聊天消息相关接口
    （main.py会在chat_messages的msg_type='transfer'时把transfer_id关联起来，
    两个动作分开发起，钱包记录和聊天气泡是两件独立但可以互相引用的事）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    from_user = data.get("from")
    to_user = data.get("to")
    amount = data.get("amount")
    message = data.get("message")
    try:
        row = create_transfer(from_user, to_user, amount, message=message, source="manual")
        if not row:
            return jsonify({"ok": False, "error": "写入失败"}), 500
        return jsonify({"ok": True, "transfer": row})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        log_error("post_wallet_transfer", e)
        return jsonify({"ok": False, "error": str(e)}), 500
