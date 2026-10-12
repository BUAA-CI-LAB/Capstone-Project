package com.example.llama.new

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
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
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.google.android.material.appbar.MaterialToolbar
import kotlinx.coroutines.Dispatchers
import com.example.llama.R
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.lang.StringBuilder

class ChatFragment : Fragment(R.layout.fragment_chat) {

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

    lateinit var engine: InferenceEngine
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

        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(requireActivity().applicationContext)
        }

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
            inputEdit.isEnabled = false

            currentConversation?.messages?.add(Message(Role.USER, userText))

            if (currentConversation?.title == "") {
                lifecycleScope.launch {
                    val END_TAG = "<|im_end|>"
                    val buffer = StringBuilder()
                    val prompt =
                        "请基于用户的提问生成一个一句话的对话标题，只需要标题本身，不要任何解释或额外文字，不超过10个字。\n" +
                            "用户提问：" + userText
                    buffer.clear()
                    var text: String = ""
                    lifecycleScope.launch(Dispatchers.Default) {
                        engine.sendUserPrompt(prompt)
                            .onCompletion {
                                withContext(Dispatchers.Main) {
                                    inputEdit.isEnabled = true
                                }
                                Log.d("ChatFragment", "title making $text")
                                text = text.replace("\"", "")
                                text = text.replace("\n", "")
                                if (text.length > 10) {
                                    text = text.take(10)
                                }
                                withContext(Dispatchers.Main) {
                                    currentConversation?.title = text
                                    title.text = text
                                    currentConversation?.let { conv ->
                                        HistoryManager.saveConversation(requireContext(), conv)
                                    }
                                    (requireActivity() as MainActivity_new).refreshHistoryMenu()
                                    engine.clearMemory()
                                }

                                lifecycleScope.launch {
                                    val bubbleView = addMessageBubble("", isUser = false)

                                    val END_TAG = "<|im_end|>"
                                    val buffer = StringBuilder()

                                    lifecycleScope.launch(Dispatchers.Default) {
                                        engine.sendUserPrompt(userText)
                                            .onCompletion {
                                                withContext(Dispatchers.Main) {
                                                    inputEdit.isEnabled = true
                                                }
                                                currentConversation?.messages?.add(Message(Role.ASSISTANT, bubbleView.text.toString()))
                                                currentConversation?.let { conv ->
                                                    HistoryManager.saveConversation(requireContext(), conv)
                                                }
                                            }.collect {
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
                                    }
                                }
                            }.collect {
                                mainHandler.post {
                                    buffer.append(it)

                                    if (buffer.isNotEmpty() && !END_TAG.startsWith(buffer.toString()) &&
                                        !buffer.toString().startsWith(END_TAG)) {
                                        val token = buffer.toString()
                                        buffer.clear()
                                        text += token
                                    }

                                    scrollView.post {
                                        scrollView.fullScroll(ScrollView.FOCUS_DOWN)
                                    }
                                }
                            }
                    }
                }
            } else {
                lifecycleScope.launch {
                    val bubbleView = addMessageBubble("", isUser = false)

                    val END_TAG = "<|im_end|>"
                    val buffer = StringBuilder()

                    lifecycleScope.launch(Dispatchers.Default) {
                        engine.sendUserPrompt(userText)
                            .onCompletion {
                                withContext(Dispatchers.Main) {
                                    inputEdit.isEnabled = true
                                }
                                currentConversation?.messages?.add(Message(Role.ASSISTANT, bubbleView.text.toString()))
                                currentConversation?.let { conv ->

                                    Log.d(
                                        "ChatFragment",
                                        "save conversation [${currentConversation!!.title}]"
                                    )
                                    HistoryManager.saveConversation(requireContext(), conv)
                                }
                            }.collect {
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
                    }
                }
            }
        }
    }

    fun openNewConversation() {
        currentConversation = Conversation(title = "")

        if (currentModel != null && currentModel != "") {
            engine.clearMemory()
        }

        title.text = "新建对话"
        messagesContainer.removeAllViews()
    }

    fun loadConversation(conversation: Conversation) {
        if (currentModel == null || currentModel == "") {
            Toast.makeText(context, "请加载模型", Toast.LENGTH_SHORT).show()
            return
        }
        currentConversation = conversation

        title.text = conversation.title
        engine.clearMemory()

        Log.d("loadConversation", conversation.title)
        messagesContainer.removeAllViews()

        conversation.messages.forEach { msg ->
            when (msg.role) {
                Role.USER -> addMessageBubble(msg.content, true)
                Role.ASSISTANT -> addMessageBubble(msg.content, false)
                Role.SYSTEM -> {}
            }
        }

        engine.prefillHistory(buildPromptFromConversation(conversation))
    }

    fun onConversationDeleted(deletedId: String) {
        if (currentConversation?.id == deletedId) {
            openNewConversation()
        }
    }

    fun buildPromptFromConversation(conversation: Conversation, maxChars: Int = 4000, isAssistant: Boolean = true): Array<String> {
        val result = mutableListOf<String>()
        var currentLength = 0

        for (msg in conversation.messages.asReversed()) {
            val line = msg.content

            if (currentLength + line.length > maxChars) break

            result.add(0, line) // 倒序插入，保持原始顺序
            currentLength += line.length
        }

        return result.toTypedArray()
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

        currentConversation?.let { engine.prefillHistory(buildPromptFromConversation(it)) }
    }
}
