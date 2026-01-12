package com.example.llama

import android.llama.cpp.LLamaAndroid
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class MainViewModel(private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()): ViewModel() {
    companion object {
        @JvmStatic
        private val NanosPerSecond = 1_000_000_000.0
    }

    private val tag: String? = this::class.simpleName

    // 1. 可观察变量
    // 1.1 就是说var <value_name> by mutableStateOf(listOf<initial_value>)的本质
    // 1.1 是声明了一个listOf类型的变量<value_name>，并且设置初始值为<initial_value>
    // 1.2 然后通过by mutableStateOf给这个变量新增一个功能，使得这个变量更新的时候，页面上显示这个变量的地方会自动刷新
    // 1.3 private set
    // 1.3 private set就是把set方法设置为私有方法
    // 1.3 不像Java中要在类的方法中专门写set和get方法然后设置public或private，这里直接在变量后面加一个private set就可以，减少代码量
    var messages by mutableStateOf(listOf("初始化..."))
        private set

    var message by mutableStateOf("")
        private set

    override fun onCleared() {
        super.onCleared()

        viewModelScope.launch {
            try {
                llamaAndroid.unload()
            } catch (exc: IllegalStateException) {
                messages += exc.message!!
            }
        }
    }

    fun send() {
        val text = message
        message = ""

        // Add to messages console.
        messages += text
        messages += ""

        viewModelScope.launch {
            llamaAndroid.send(text)
                .catch {
                    Log.e(tag, "send() failed", it)
                    messages += it.message!!
                }
                .collect { messages = messages.dropLast(1) + (messages.last() + it) }
        }
    }

    fun load(pathToModel: String) {
        viewModelScope.launch {
            try {
                llamaAndroid.load(pathToModel)
                messages += "已加载 $pathToModel"
            } catch (exc: IllegalStateException) {
                Log.e(tag, "load() failed", exc)
                messages += exc.message!!
            }
        }
    }

    fun updateMessage(newMessage: String) {
        message = newMessage
    }

    // 1. 追加messages
    // 1.1 messages是listOf(<String>)
    fun log(message: String) {
        messages += message
    }
}
