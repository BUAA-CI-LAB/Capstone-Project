package com.example.llama

import android.llama.cpp.LLamaAndroid
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.widget.Toolbar
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.example.llama.new.Conversation
import com.example.llama.new.HistoryManager
import com.example.llama.new.MainActivity_new
import com.example.llama.new.Message
import com.example.llama.new.Role
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.navigation.NavigationView
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import org.w3c.dom.Text
import java.lang.StringBuilder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ChatFragment : Fragment(R.layout.fragment_chat) {

//    val viewModel = ViewModelProvider(requireActivity())[MainViewModel_new::class.java]
    lateinit var menuItem: MenuItem

    var currentModel: String? = null

    // 当前对话
    private var currentConversation: Conversation? = null

    // UI 元素
    private lateinit var inputEdit: EditText
    private lateinit var sendBtn: Button
    private lateinit var scrollView: ScrollView
    private lateinit var messagesContainer: LinearLayout
    private lateinit var title: TextView
    val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()

    // 对话历史
    private val history = mutableListOf<String>()

    // 后台线程池 & 主线程 handler
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {

        val toolbar = view.findViewById<MaterialToolbar>(R.id.toolbar)
        title = view.findViewById<TextView>(R.id.conversation_title)
        menuItem = toolbar.menu.findItem(R.id.action_model)

        val navigationButton: ImageButton = view.findViewById(R.id.navigation_button)

        navigationButton.setOnClickListener {
            (requireActivity() as MainActivity_new).openDrawer()
        }

        toolbar.title = ""

        menuItem.title = "选择模型"

        toolbar.setOnMenuItemClickListener {
            if (it.itemId == R.id.action_model) {
                ModelSelectBottomSheet { action ->
                    if (action == ModelSelectBottomSheet.Action.DOWNLOAD) {
                        (requireActivity() as MainActivity_new).showModelDownload()
                    }
                }.show(parentFragmentManager, "model_select")
                true
            } else false
        }

        currentConversation = Conversation(title = "")

        // 绑定 UI
        inputEdit = view.findViewById(R.id.inputEdit)
        sendBtn = view.findViewById(R.id.sendBtn)
        scrollView = view.findViewById(R.id.scrollView)
        messagesContainer = view.findViewById(R.id.messagesContainer)

        val systemMessage = Message(Role.SYSTEM, "你是一个智能助手。")
        currentConversation?.messages?.add(systemMessage)

        sendBtn.setOnClickListener {
            if (currentModel == null || currentModel == "") {
                Toast.makeText(context, "请加载模型", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            var userText = inputEdit.text.toString().trim()
            if (userText.isEmpty()) {
                return@setOnClickListener
            }

            // 显示用户消息（右侧气泡）
            addMessageBubble(userText, isUser = true)
            history.add("User: $userText")

            inputEdit.text.clear()

//            if (currentConversation?.title == "") {
//                lifecycleScope.launch {
//                    val END_TAG = "<|im_end|>"
//                    val buffer = StringBuilder()
//                    val prompt =
//                        "user\n你是一个标题总结智能助手，根据用户的提问，为这一次对话总结一个简短的标题，标题最多不超过10个字；要求：直接输出内容，不需要其它的符号;" +
//                            "用户提问：\"" + userText + "\"" +
//                            "\n<|im_start|>assistant"
//                    buffer.clear()
//                    var text: String = ""
//                    llamaAndroid.send(prompt)
//                        .catch {
//                            Log.e("TAG", "send() failed", it)
//                        }
//                        .collect {
//                            mainHandler.post {
//                                buffer.append(it)
//
//                                if (buffer.isNotEmpty() && !END_TAG.startsWith(buffer.toString()) &&
//                                    !buffer.toString().startsWith(END_TAG)
//                                ) {
//                                    val token = buffer.toString()
//                                    buffer.clear()
//                                    text += token
//                                }
//                            }
//                        }
//                    text = text.replace("\"", "")
//                    text = text.replace("\n", "")
//                    if (text.length > 10) {
//                        text = text.take(10)
//                    }
//                    currentConversation?.title = text
//                    title.text = text
//
//                    currentConversation?.let { conv ->
//                        HistoryManager.saveConversation(requireContext(), conv)
//                    }
//                    (requireActivity() as MainActivity_new).refreshHistoryMenu()
//                }
//            }

            currentConversation?.messages?.add(Message(Role.USER, userText))

//            val prompt = currentConversation?.let { conv ->
//                buildPromptFromConversation(conv, maxChars = 4000)
//            } ?: userText

            val prompt = "<|im_start|>user\n$userText<|im_start|>assistant\n"

            val useBackend = true
            if (useBackend) {
                lifecycleScope.launch {
                    val bubbleView = addMessageBubble("", isUser = false)

                    val END_TAG = "<|im_end|>"
                    val buffer = StringBuilder()

                    llamaAndroid.send2(prompt)
                        .catch {
                            Log.e("TAG", "send() failed", it)
                        }
                        .collect {
                            mainHandler.post {
                                buffer.append(it)

                                if (buffer.isNotEmpty() && !END_TAG.startsWith(buffer.toString()) &&
                                    !buffer.toString().startsWith(END_TAG)) {
                                    val token = buffer.toString()
                                    buffer.clear()
                                    bubbleView.text = bubbleView.text.toString() + token
                                }

                                scrollView.post {
                                    scrollView.fullScroll(ScrollView.FOCUS_DOWN)
                                }
                            }
                        }
                    currentConversation?.messages?.add(Message(Role.ASSISTANT, bubbleView.text.toString()))

                    Log.d("currentConversation.title", currentConversation!!.title)

                    Log.d("output", bubbleView.text.toString())
                    currentConversation?.let { conv ->
                        HistoryManager.saveConversation(requireContext(), conv)
                    }
                }
            }
        }
    }

    fun openNewConversation() {
        currentConversation = Conversation(title = "")

        val systemMessage = Message(Role.SYSTEM, "你是一个智能助手。")
        currentConversation?.messages?.add(systemMessage)

        title.text = "新建对话"
        messagesContainer.removeAllViews()
    }

    fun loadConversation(conversation: Conversation) {
        currentConversation = conversation

        title.text = conversation.title

        Log.d("loadConversation", conversation.title)
        messagesContainer.removeAllViews()

        conversation.messages.forEach { msg ->
            when (msg.role) {
                Role.USER -> addMessageBubble(msg.content, true)
                Role.ASSISTANT -> addMessageBubble(msg.content, false)
                Role.SYSTEM -> {}
            }
        }
    }

    fun onConversationDeleted(deletedId: String) {
        if (currentConversation?.id == deletedId) {
            openNewConversation()
        }
    }

    fun buildPromptFromConversation(conversation: Conversation, maxChars: Int = 4000, isAssistant: Boolean = true): String {
        val sb = StringBuilder()

        if (isAssistant) {
            sb.append("<|im_start|>assistant\n")
        }
        // 倒序添加消息，保证最新消息保留
        for (msg in conversation.messages.asReversed()) {
            val line = when (msg.role) {
                Role.USER -> "user\n ${msg.content}\n"
                Role.ASSISTANT -> "assistant\n ${msg.content}\n"
                Role.SYSTEM -> "system\n ${msg.content}\n"
            }

            if (sb.length + line.length > maxChars) break
            sb.insert(0, line) // 倒序插入，保持顺序
        }
//        Log.d("aa", sb.toString())

        return sb.toString()
    }

    /**
     * 添加聊天气泡
     */
    private fun addMessageBubble(text: String?, isUser: Boolean): TextView {
        val outer = LinearLayout(requireActivity()).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).also {
                it.topMargin = dpToPx(6)
                it.bottomMargin = dpToPx(6)
                it.marginStart = dpToPx(4)
                it.marginEnd = dpToPx(4)
            }
        }

        val bubble = TextView(requireActivity()).apply {
            setText(text)
            textSize = 15f
            setLineSpacing(0f, 1.1f)
            setPadding(dpToPx(12), dpToPx(8), dpToPx(12), dpToPx(8))
            maxWidth = (resources.displayMetrics.widthPixels * 0.75).toInt()
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            if (isUser) {
                setBackgroundResource(R.drawable.bubble_right)
                setTextColor(0xFFFFFFFF.toInt())
            } else {
                setBackgroundResource(R.drawable.bubble_left)
                setTextColor(0xFF111111.toInt())
            }
        }

        if (isUser) {
            val spacer = LinearLayout(requireActivity()).apply {
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            outer.addView(spacer)
            outer.addView(bubble)
        } else {
            outer.addView(bubble)
            val spacer = LinearLayout(requireActivity()).apply {
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            outer.addView(spacer)
        }

        messagesContainer.addView(outer)
        scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }

        return bubble
    }

    private fun dpToPx(dp: Int): Int {
        val scale = resources.displayMetrics.density
        return (dp * scale + 0.5f).toInt()
    }



    fun changeModel(name: String) {
        menuItem.title = name
        currentModel = name
    }
}
