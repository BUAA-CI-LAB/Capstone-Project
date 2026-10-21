package com.example.llama.new

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

object HistoryManager {

    private const val PREF_NAME = "conversation_history"
    private const val KEY_HISTORY = "history"

    fun saveConversation(context: Context, conversation: Conversation) {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val all = loadConversations(context).toMutableList()

        val existingIndex = all.indexOfFirst { it.id == conversation.id }
        if (existingIndex >= 0) {
            all[existingIndex] = conversation
        } else {
            all.add(conversation)
        }

        // 序列化整个列表为 JSONArray
        val jsonArray = JSONArray()
        all.forEach { conv ->
            jsonArray.put(JSONObject(conversationToJson(conv)))
        }

        prefs.edit().putString(KEY_HISTORY, jsonArray.toString()).apply()
    }

    fun loadConversations(context: Context): List<Conversation> {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val str = prefs.getString(KEY_HISTORY, "[]") ?: "[]"

        val list = mutableListOf<Conversation>()
        val array = JSONArray(str)
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            val conv = conversationFromJson(obj.toString())
            list.add(conv)
        }
        return list
    }

    fun clearHistory(context: Context) {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        prefs.edit().remove(KEY_HISTORY).apply()
    }

    fun deleteConversation(context: Context, conversationId: String) {
        val prefs = context.getSharedPreferences("conversation_history", Context.MODE_PRIVATE)
        val all = loadConversations(context).toMutableList()

        val newList = all.filter { it.id != conversationId }

        val jsonArray = JSONArray()
        newList.forEach { conv ->
            jsonArray.put(JSONObject(conversationToJson(conv)))
        }

        prefs.edit().putString("history", jsonArray.toString()).apply()
    }
}
