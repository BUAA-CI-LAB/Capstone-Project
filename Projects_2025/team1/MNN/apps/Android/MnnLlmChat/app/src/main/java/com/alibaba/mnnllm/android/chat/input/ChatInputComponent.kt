// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat.input

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.speech.tts.TextToSpeech
import android.text.Editable
import android.text.TextUtils
import android.text.TextWatcher
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.chat.input.AttachmentPickerModule.AttachmentType
import com.alibaba.mnnllm.android.chat.input.AttachmentPickerModule.ImagePickCallback
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.chat.input.VoiceRecordingModule.VoiceRecordingListener
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.utils.KeyboardUtils
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.utils.Permissions.REQUEST_RECORD_AUDIO_PERMISSION
import java.util.Date
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import org.json.JSONObject
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService

class ChatInputComponent(
    private val chatActivity: ChatActivity,
    private val binding: ActivityChatBinding,
    modelId:String,
    modelName: String,
) {
    private var currentModelId: String = modelId
    private var currentModelName: String = modelName
    private var onStopGenerating: (() -> Unit)? = null
    private var onThinkingModeChanged: ((Boolean) -> Unit)? = null
    private var onAudioOutputModeChanged: ((Boolean) -> Unit)? = null
    private var onSendMessage: ((ChatDataItem) -> Unit)? = null
    private lateinit var editUserMessage: EditText
    private var buttonSend: ImageView = binding.btnSend
    private var buttonAudio: ImageView = binding.btnAudio
    private lateinit var imageMore: ImageView
    var attachmentPickerModule: AttachmentPickerModule? = null
    private lateinit var voiceRecordingModule: VoiceRecordingModule
    private var currentUserMessage: ChatDataItem? = null
    private var buttonSwitchVoice: View? = null

    init {
        buttonSend.setEnabled(false)
        buttonSend.setOnClickListener { handleSendClick() }
        buttonAudio.setOnClickListener { handleAudioClick() }
        setupEditText()
        setupAttachmentPickerModule()
        setupVoiceRecordingModule()
    }



    private lateinit var recognizer: Recognizer
    private lateinit var speechService: SpeechService
    private lateinit var tts: TextToSpeech
    private lateinit var tvAsrText: TextView
    private fun startListening() {
        Log.d("hygebra", "startListening")
        speechService = SpeechService(recognizer, 16000.0f)
        Log.d("hygebra", "startListening SpeechService ended")

        buttonAudio.setImageResource(R.drawable.ic_audio_file)
        buttonAudio.setBackgroundColor(
            ContextCompat.getColor(chatActivity, R.color.colorGreen)
        )
        editUserMessage.setHint("开始说话，默认说话时间10秒")
        speechService.startListening(object : RecognitionListener {

            override fun onPartialResult(hypothesis: String?) {
                val partial = JSONObject(hypothesis).optString("partial").replace(" ", "")
                if (partial.isNotBlank()) {
                    editUserMessage.setText(partial)
                    Log.d("hygebra", partial)
                }
                Log.d("hygebra", partial)
            }

            override fun onResult(hypothesis: String) {
                val partial = JSONObject(hypothesis).optString("partial").replace(" ", " ")
                if (partial.isNotBlank()) {
                    editUserMessage.setText(partial)
                    Log.d("hygebra", partial)
                }
                Log.d("hygebra", partial)
            }

            override fun onFinalResult(hypothesis: String) {
                val text = JSONObject(hypothesis).optString("text").replace(" ", "")
                if (text.isNotBlank()) {
                    editUserMessage.setText(text)
                    Log.d("hygebra", text)
                }
                Log.d("hygebra", text)
            }

            override fun onError(e: Exception) {
                e.printStackTrace()
            }

            override fun onTimeout() {
                Log.d("hygebra", "Speech timeout")
                buttonAudio.setImageResource(R.drawable.ic_audio)
                buttonAudio.setBackgroundColor(
                    ContextCompat.getColor(chatActivity, R.color.color_background)
                )
                speechService.stop()
            }
        }, 5000)
    }


    fun startAsr() {
//        tts = TextToSpeech(this) {
//            tts.language = Locale.CHINESE
//        }
        Log.d("hygebra", "startAsr")
        StorageService.unpack(
            chatActivity,
            "vosk-model-cn-0.22",   // assets 下的目录名
            "vosk-model-cn-0.22",      // 解压到 app 私有目录
            { model ->
                recognizer = Recognizer(model, 16000.0f)
                startListening()
            },
            { e ->
                Log.e("VOSK", "Model unpack failed", e)
            }
        )
        Log.d("hygebra", "unpack ended")
    }

    private fun requestAudioPermission_and_Start() {
        if (ContextCompat.checkSelfPermission(
                chatActivity,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                chatActivity,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                1001
            )
        } else {
            try{
                startAsr()
            }catch (e:Exception){
                e.printStackTrace()
                buttonAudio.setImageResource(R.drawable.ic_audio)
                buttonAudio.setBackgroundColor(
                    ContextCompat.getColor(chatActivity, R.color.color_background)
                )
                editUserMessage.setHint("语音模型     加载失败")
            }
        }
    }



    /**
     * Update the model name and refresh related UI components
     */
    fun updateModel(newModelId:String, newModelName: String) {
        val oldModelId = currentModelId
        currentModelId = newModelId
        val oldModelName = currentModelName
        currentModelName = newModelName
        
        // Update attachment picker if model capabilities changed
        if (ModelTypeUtils.isVisualModel(oldModelId) != ModelTypeUtils.isVisualModel(newModelId) ||
            ModelTypeUtils.isAudioModel(oldModelId) != ModelTypeUtils.isAudioModel(newModelId)) {
            setupAttachmentPickerModule()
        }
        // Update voice recording module
        voiceRecordingModule.updateModel(newModelName)
        
        // Update voice button visibility
        updateVoiceButtonVisibility()
    }

    private fun handleSendClick() {
        Log.d(
            TAG,
            "handleSendClick isGenerating : ${chatActivity.isGenerating}"
        )
        if (chatActivity.isGenerating) {
            this.onStopGenerating?.invoke()
        } else {
            sendUserMessage()
        }
    }

    private fun handleAudioClick() {
        Log.d("hygebra", "handleAudioClick")
        editUserMessage.setHint("正在加载语音组件vosk-model-cn-0.22")
        buttonAudio.setImageResource(R.drawable.ic_audio_off)
        buttonAudio.setBackgroundColor(
            ContextCompat.getColor(chatActivity, R.color.colorRed)
        )
        requestAudioPermission_and_Start()
    }

    private fun setupEditText() {
        editUserMessage = binding.etMessage
        editUserMessage.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence, start: Int, count: Int, after: Int) {
            }

            override fun onTextChanged(s: CharSequence, start: Int, before: Int, count: Int) {
            }

            override fun afterTextChanged(s: Editable) {
                updateSenderButton()
                updateVoiceButtonVisibility()
            }
        })
    }

    fun updateSenderButton() {
        var enabled = true
        if (chatActivity.isLoading) {
            enabled = false
        } else if (currentUserMessage == null && TextUtils.isEmpty(editUserMessage.text.toString())) {
            enabled = false
        }
        if (chatActivity.isGenerating) {
            enabled = true
        }
        buttonSend.isEnabled = enabled
        buttonSend.setImageResource(if (!chatActivity.isGenerating) R.drawable.button_send else R.drawable.ic_stop)
    }

    private fun sendUserMessage() {
        if (!buttonSend.isEnabled) {
            return
        }
        val inputString = editUserMessage.text.toString().trim { it <= ' ' }
        if (currentUserMessage == null) {
            currentUserMessage = ChatDataItem(ChatViewHolders.USER)
        }
        currentUserMessage!!.text = inputString
        currentUserMessage!!.time = chatActivity.dateFormat!!.format(Date())
        editUserMessage.setText("")
        KeyboardUtils.hideKeyboard(editUserMessage)
        this.onSendMessage?.let { it(currentUserMessage!!) }
        if (attachmentPickerModule != null) {
            attachmentPickerModule!!.clearInput()
            attachmentPickerModule!!.hideAttachmentLayout()
        }
        currentUserMessage = null
    }

    private fun updateVoiceButtonVisibility() {
        if (!ModelTypeUtils.isAudioModel(currentModelId)) {
            return
        }
        var visible = true
        if (!ModelTypeUtils.isAudioModel(currentModelId)) {
            visible = false
        } else if (chatActivity.isGenerating) {
            visible = false
        } else if (currentUserMessage != null) {
            visible = false
        } else if (!TextUtils.isEmpty(editUserMessage.text.toString())) {
            visible = false
        }
        buttonSwitchVoice!!.visibility =
            if (visible) View.VISIBLE else View.GONE
    }

    private fun setupAttachmentPickerModule() {
        imageMore = binding.btPlus
        buttonSwitchVoice = binding.btSwitchAudio
        if (!ModelTypeUtils.isVisualModel(currentModelId) && !ModelTypeUtils.isAudioModel(currentModelId)) {
            imageMore.setVisibility(View.GONE)
            return
        }
        attachmentPickerModule = AttachmentPickerModule(chatActivity)
        attachmentPickerModule!!.setOnImagePickCallback(object : ImagePickCallback {
            override fun onAttachmentPicked(imageUri: Uri?, audio: AttachmentType?) {
                imageMore.setVisibility(View.GONE)
                updateVoiceButtonVisibility()
                currentUserMessage = ChatDataItem(ChatViewHolders.USER)
                when (audio) {
                    AttachmentType.Audio -> {
                        currentUserMessage!!.audioUri = imageUri
                    }
                    AttachmentType.Video -> {
                        currentUserMessage!!.videoUri = imageUri
                    }
                    else -> {
                        currentUserMessage!!.imageUri = imageUri
                    }
                }
                updateSenderButton()
            }

            override fun onAttachmentRemoved() {
                currentUserMessage = null
                imageMore.setVisibility(View.VISIBLE)
                updateSenderButton()
                updateVoiceButtonVisibility()
            }

            override fun onAttachmentLayoutShow() {
                imageMore.setImageResource(R.drawable.ic_bottom)
            }

            override fun onAttachmentLayoutHide() {
                imageMore.setImageResource(R.drawable.ic_plus)
            }
        })
        imageMore.setOnClickListener {
            voiceRecordingModule.exitRecordingMode()
            attachmentPickerModule!!.toggleAttachmentVisibility()
        }
    }

    private fun setupVoiceRecordingModule() {
        voiceRecordingModule = VoiceRecordingModule(chatActivity)
        voiceRecordingModule.setOnVoiceRecordingListener(object : VoiceRecordingListener {
            override fun onEnterRecordingMode() {
//                binding.btnToggleThinking.visibility = View.GONE
                editUserMessage.visibility = View.GONE
                KeyboardUtils.hideKeyboard(editUserMessage)
                if (attachmentPickerModule != null) {
                    attachmentPickerModule!!.hideAttachmentLayout()
                }
                editUserMessage.visibility = View.GONE
            }

            override fun onLeaveRecordingMode() {
                val extraTags = ModelListManager.getExtraTags(currentModelId)
                if (ModelTypeUtils.isSupportThinkingSwitchByTags(extraTags)) {
//                    binding.btnToggleThinking.visibility = View.VISIBLE
                }
                binding.btnSend.visibility = View.VISIBLE
                editUserMessage.visibility = View.VISIBLE
                editUserMessage.requestFocus()
                KeyboardUtils.showKeyboard(editUserMessage)
            }

            override fun onRecordSuccess(duration: Float, recordingFilePath: String?) {
                val chatDataItem = ChatDataItem.createAudioInputData(
                    chatActivity.dateFormat!!.format(Date()),
                    "",
                    recordingFilePath!!,
                    duration
                )
                this@ChatInputComponent.onSendMessage?.let { it(chatDataItem) }
            }

            override fun onRecordCanceled() {
            }
        })
        voiceRecordingModule!!.setup(chatActivity.isAudioModel)
    }

    fun setOnSendMessage(onSendMessage: (ChatDataItem)->Unit) {
        this.onSendMessage = onSendMessage
    }

    fun setOnThinkingModeChanged(onThinkingModeChanged: (Boolean)->Unit) {
        this.onThinkingModeChanged = onThinkingModeChanged
    }

    fun setOnAudioOutputModeChanged(onAudioOutputChanged: (Boolean)->Unit) {
        this.onAudioOutputModeChanged = onAudioOutputChanged
    }

    fun setIsGenerating(generating: Boolean) {
        updateSenderButton()
        updateVoiceButtonVisibility()
    }

    fun handleResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (attachmentPickerModule != null && attachmentPickerModule!!.canHandleResult(requestCode)) {
            attachmentPickerModule?.onActivityResult(requestCode, resultCode, data)
        }
    }

    fun onLoadingStatesChanged(loading: Boolean) {
        this.updateSenderButton()
        if (!loading && ModelTypeUtils.isAudioModel(currentModelId)) {
            voiceRecordingModule.onEnabled()
        }
    }

    fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                voiceRecordingModule.handlePermissionAllowed()
            } else {
                voiceRecordingModule.handlePermissionDenied()
            }
        } else if (attachmentPickerModule != null && 
                   requestCode == AttachmentPickerModule.REQUEST_CODE_CAMERA_PERMISSION) {
            attachmentPickerModule!!.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    }

    fun setOnStopGenerating(onStopGenerating: () -> Unit) {
        this.onStopGenerating = onStopGenerating
    }

}