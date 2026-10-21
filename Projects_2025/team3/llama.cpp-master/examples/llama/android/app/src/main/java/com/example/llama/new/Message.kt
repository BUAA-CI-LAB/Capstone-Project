package com.example.llama.new

import org.json.JSONArray
import org.json.JSONObject

data class Message(
    val role: Role,
    val content: String,
    val timestamp: Long = System.currentTimeMillis()
)

enum class Role { USER, ASSISTANT, SYSTEM }

data class Conversation(
    val id: String = System.currentTimeMillis().toString(), // 唯一 id
    var title: String,
    val messages: MutableList<Message> = mutableListOf(),
    val lastUpdate: Long = System.currentTimeMillis()
)
fun conversationToJson(conv: Conversation): String {
    val json = JSONObject()
    json.put("id", conv.id)
    json.put("title", conv.title)
    json.put("lastUpdate", conv.lastUpdate)

    val msgsArray = JSONArray()
    for (msg in conv.messages) {
        val msgObj = JSONObject()
        msgObj.put("role", msg.role.name)
        msgObj.put("content", msg.content)
        msgObj.put("timestamp", msg.timestamp)
        msgsArray.put(msgObj)
    }

    json.put("messages", msgsArray)
    return json.toString()
}

fun conversationFromJson(jsonStr: String): Conversation {
    val json = JSONObject(jsonStr)
    val id = json.getString("id")
    val title = json.getString("title")
    val lastUpdate = json.getLong("lastUpdate")

    val messages = mutableListOf<Message>()
    val msgsArray = json.getJSONArray("messages")
    for (i in 0 until msgsArray.length()) {
        val msgObj = msgsArray.getJSONObject(i)
        messages.add(
            Message(
                role = Role.valueOf(msgObj.getString("role")),
                content = msgObj.getString("content"),
                timestamp = msgObj.getLong("timestamp")
            )
        )
    }

    return Conversation(id = id, title = title, messages = messages, lastUpdate = lastUpdate)
}
