# ============================================================
# 表情包 (Stickers)
# ============================================================
# 独立模块。表情包是"素材库"：这里只管素材本身的增删查，不管
# "在聊天里发送一个表情包"这个动作本身——发送时main.py的chat_send
# 直接往chat_messages插入一条msg_type='sticker'的记录，引用这里的sticker id，
# 不需要经过这个模块的路由（同一个道理：moments.py不管理"图片"本身，
# 只是发动态时可以带一个image_url）。
#
# 依赖core.py的公共基础设施（包括图片上传封装），不反向依赖main.py。
#
# 建表SQL见 sql/schema_additions.sql 里的 stickers 部分。

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


def load_stickers():
    """读全部表情包，按上传时间倒序（最近上传的排在选择器最前面，符合"最近常用"的直觉，
    表情包数量级不大，不需要分页）。"""
    try:
        rows = _supabase_request(
            "GET", "stickers",
            params={"select": "id,uploaded_by,image_url,name,created_at", "order": "created_at.desc"}
        )
        return rows or []
    except Exception as e:
        log_error("load_stickers", e)
        return []


def add_sticker_row(uploaded_by, image_url, name=None):
    body = {
        "uploaded_by": uploaded_by,
        "image_url": image_url,
        "name": (name or "").strip() or None,
        "created_at": datetime.now().isoformat(),
    }
    rows = _supabase_request("POST", "stickers", json_body=body, headers_extra={"Prefer": "return=representation"})
    return rows[0] if rows else None


def get_sticker_row(sticker_id):
    rows = _supabase_request(
        "GET", "stickers",
        params={"select": "id,uploaded_by,image_url,name,created_at", "id": f"eq.{sticker_id}", "limit": 1}
    )
    return rows[0] if rows else None


def delete_sticker_row(sticker_id):
    """删除一个表情包素材：先拿到它的image_url去清理Storage里的文件，再删数据库行。
    注意：已经发送到聊天记录里的历史消息不受影响——chat_messages.extra里
    冗余存了一份image_url快照（见schema注释），历史消息该长什么样还是什么样，
    只是这个表情包以后不能再被选中发送了。"""
    row = get_sticker_row(sticker_id)
    if row and row.get("image_url"):
        delete_image_from_storage(row["image_url"])
    _supabase_request("DELETE", "stickers", params={"id": f"eq.{sticker_id}"})


# ---- API 路由 ----

@app.route("/api/stickers", methods=["GET"])
def get_stickers():
    """拉取全部表情包素材，供前端表情选择面板渲染。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return jsonify({"ok": True, "stickers": load_stickers()})


@app.route("/api/stickers/upload", methods=["POST"])
def upload_sticker():
    """上传一个新表情包。multipart/form-data，字段：
    file（图片文件，必填）、uploaded_by（'user'或'charon'，默认'user'）、name（可选标签）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "缺少file文件"}), 400
    uploaded_by = request.form.get("uploaded_by", "user")
    name = request.form.get("name")
    if uploaded_by not in ("user", "charon"):
        return jsonify({"ok": False, "error": "uploaded_by必须是user或charon"}), 400
    try:
        image_url = upload_image_to_storage(file.read(), file.filename, folder="stickers")
        row = add_sticker_row(uploaded_by, image_url, name=name)
        if not row:
            return jsonify({"ok": False, "error": "写入失败"}), 500
        return jsonify({"ok": True, "sticker": row})
    except Exception as e:
        log_error("upload_sticker", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/stickers/delete", methods=["POST"])
def delete_sticker():
    """删除一个表情包素材。body: {"id": "..."}"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    sticker_id = data.get("id")
    if not sticker_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400
    try:
        delete_sticker_row(sticker_id)
        return jsonify({"ok": True, "deleted_id": sticker_id})
    except Exception as e:
        log_error("delete_sticker", e)
        return jsonify({"ok": False, "error": str(e)}), 500
