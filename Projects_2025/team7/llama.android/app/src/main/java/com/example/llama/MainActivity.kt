package com.example.llama

import android.Manifest
import android.app.ActivityManager
import android.app.DownloadManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.text.format.Formatter
import android.media.AudioRecord
import android.media.AudioFormat
import android.media.MediaRecorder
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener as VoskRecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService
import org.json.JSONObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import com.example.llama.ui.theme.LlamaAndroidTheme
import java.io.File
import androidx.compose.foundation.layout.size
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.OutlinedTextFieldDefaults

class MainActivity(
    activityManager: ActivityManager? = null,
    downloadManager: DownloadManager? = null,
    clipboardManager: ClipboardManager? = null,
): ComponentActivity() {
    private val tag: String? = this::class.simpleName

    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }

    private val viewModel: MainViewModel by viewModels()

    // 麦克风权限
    // requestPermissionLauncher是一个工具变量，是registerForActivityResult<String>类的实例化
    // 自带.launch()方法，这个方法接受一个参数，这个参数必须是ActivityResultContracts.RequestPermission()类型，比如Manifest.permission.RECORD_AUDIO就是
    // 然后requestPermissionLauncher的.launch()方法根据传入的权限类型去弹出弹窗，根据用户是否同意授予权限，去执行回调函数
    // 这个回调函数的形式是 参数 -> {执行代码}，参数其实就是用户是否同意授予权限的一个Boolean值
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            viewModel.log("麦克风权限已授予")
        } else {
            viewModel.log("麦克风权限被拒绝")
        }
    }

    // Vosk 语音识别相关
    private var model: Model? = null
    private var speechService: SpeechService? = null

    fun checkAndRequestPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> { }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    // Get a MemoryInfo object for the device's current memory status.
    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    // 1. MainActivity类是程序打开之后运行的第一个类
    // 1. 并且打开之后对MainActivity实例化之后，执行OnCreate()方法
    override fun onCreate(savedInstanceState: Bundle?) {
        // 1.1 不需要关心
        super.onCreate(savedInstanceState)

        // 1.2 获取麦克风权限
        checkAndRequestPermission()

        // 1.2 不需要关心
        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        // 1.3 获取内存情况，并追加在viewModel的messages中
        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)
        viewModel.log("空闲内存/总内存: $free / $total")
        viewModel.log("模型下载目录: ${getExternalFilesDir(null)}")

        // 1.3.1 初始化Vosk模型
        initVoskModel()

        // 1.4 需要下载的模型列表
        val extFilesDir = getExternalFilesDir(null)
        val models = listOf(
            Downloadable(
                "TinyLlama 1.1B (int8, 1.17 GB)",
                Uri.parse("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true"),  // 本地路径也可以写成 file://
                File(extFilesDir, "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")  // 保存/使用路径
            ),
            Downloadable(
                "Qwen3 4B (int4, 2.38 GB)",
                Uri.parse("https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_0.gguf?download=true"),  // 本地路径也可以写成 file://
                File(extFilesDir, "Qwen3-4B-Q4_0.gguf")  // 保存/使用路径
            ),
            // Downloadable(
            //     "Qwen2.5 0.5B (fp16, 1.27 GB)",
            //     Uri.parse("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true"),  // 本地路径也可以写成 file://
            //     File(extFilesDir, "qwen2.5-0.5b-instruct-fp16.gguf")  // 保存/使用路径
            // )
        )

        // 1.5 SetContent是一个函数
        // 1.5.1 程序入口是MainActivity，并且一进来实例化之后会马上执行OnCreate()
        // 1.5.1 一般MainActivity的OnCreate中都会有调用SetContent，因为SetContent之前的代码都是在创建或者声明变量，然后获取一些值，只有SetContent是真正在手机屏幕上展示内容
        // 1.5.2 func在声明的时候要求传入3个参数func(arg1, arg2, arg3){}，arg3是一个函数，然后在Kotlin的语法糖中，可以写成func(arg1, arg2){arg3}，原因是arg3的代码可能很长，这样会更好看
        // 1.5.2 如果func在声明的时候只要求传入1个参数，且这个参数是函数，就可以直接写成func{arg1}
        // 1.5.2 如果以上情况中所有参数都不是函数，只是一个变量（比如int，double或者某个类的实例），就不可以这样
        // 1.5.2 SetContent(arg1){}中，arg1就是一个函数，所以可以写成SetContent{arg1}

        // 1.5.3 SetContent(arg1:lambda){}，所以写成SetContent{arg1}
        // 1.5.3 LlamaAndroidTheme(arg1:lambda){}，所以写成LlamaAndroidTheme{arg1}
        // 1.5.3 Surface(arg1, arg2, arg3:lambda){}，所以写成Surface(arg1, arg2){arg3}
        setContent {
            LlamaAndroidTheme {
                // A surface container using the 'background' color from the theme
                // 这个Surface的color是对话框背景的color，所以把colorScheme.background设置成白色
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    // 1.6 arg3:lambda
                    // 1.6.1 需要传入viewModel，剪切板，下载器，模型列表
                    // 1.6.2 arg3的具体声明代码在本类之后
                    MainCompose(
                        viewModel,
                        clipboardManager,
                        downloadManager,
                        models,
                        ::startVoskRecognition,
                        ::stopVoskRecognition
                    )
                }
            }
        }
    }

    // 初始化Vosk模型
    private fun initVoskModel() {
        Thread {
            try {
                // viewModel.log("开始加载Vosk模型...")
                // 将assets中的模型解压到应用的缓存目录
                val modelDir = File(cacheDir, "model-en")

                // 如果缓存目录中没有模型，则从assets复制
                if (!modelDir.exists()) {
                    // viewModel.log("首次加载，正在复制模型文件...")
                    modelDir.mkdirs()

                    // 复制assets中的所有文件到缓存目录
                    copyAssetFolder("model-en", modelDir.absolutePath)
                }

                // 使用缓存目录中的模型创建Model对象
                model = Model(modelDir.absolutePath)
                // viewModel.log("Vosk模型加载成功")
            } catch (e: Exception) {
                viewModel.log("Vosk模型加载失败: ${e.message}")
                e.printStackTrace()
            }
        }.start()
    }

    // 递归复制assets文件夹到目标路径
    private fun copyAssetFolder(assetPath: String, targetPath: String) {
        try {
            val assetList = assets.list(assetPath) ?: return

            if (assetList.isEmpty()) {
                // 这是一个文件，复制它
                assets.open(assetPath).use { input ->
                    File(targetPath).outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            } else {
                // 这是一个目录，递归复制
                val targetDir = File(targetPath)
                targetDir.mkdirs()

                for (asset in assetList) {
                    val assetFilePath = if (assetPath.isEmpty()) asset else "$assetPath/$asset"
                    val targetFilePath = "$targetPath/$asset"
                    copyAssetFolder(assetFilePath, targetFilePath)
                }
            }
        } catch (e: Exception) {
            viewModel.log("复制资源文件失败: ${e.message}")
            e.printStackTrace()
        }
    }

    // 开始Vosk语音识别
    private fun startVoskRecognition(onResult: (String) -> Unit, onPartialResult: (String) -> Unit) {
        if (model == null) {
            viewModel.log("模型未加载，无法开始识别")
            return
        }

        try {
            val recognizer = Recognizer(model, 16000.0f)
            speechService = SpeechService(recognizer, 16000.0f)

            speechService?.startListening(object : VoskRecognitionListener {
                override fun onPartialResult(hypothesis: String?) {
                    hypothesis?.let {
                        try {
                            val jsonObject = JSONObject(it)
                            val partial = jsonObject.optString("partial", "")
                            if (partial.isNotEmpty()) {
                                onPartialResult(partial)
                            }
                        } catch (e: Exception) {
                            viewModel.log("解析部分结果失败: ${e.message}")
                        }
                    }
                }

                override fun onResult(hypothesis: String?) {
                    hypothesis?.let {
                        try {
                            val jsonObject = JSONObject(it)
                            val text = jsonObject.optString("text", "")
                            if (text.isNotEmpty()) {
                                onResult(text)
                            }
                        } catch (e: Exception) {
                            viewModel.log("解析最终结果失败: ${e.message}")
                        }
                    }
                }

                override fun onFinalResult(hypothesis: String?) {
                    hypothesis?.let {
                        try {
                            val jsonObject = JSONObject(it)
                            val text = jsonObject.optString("text", "")
                            if (text.isNotEmpty()) {
                                onResult(text)
                            }
                        } catch (e: Exception) {
                            viewModel.log("解析最终结果失败: ${e.message}")
                        }
                    }
                }

                override fun onError(exception: Exception?) {
                    viewModel.log("识别错误: ${exception?.message}")
                }

                override fun onTimeout() {
                    viewModel.log("识别超时")
                }
            })

            // viewModel.log("开始语音识别...")
        } catch (e: Exception) {
            viewModel.log("启动识别失败: ${e.message}")
        }
    }

    // 停止Vosk语音识别
    private fun stopVoskRecognition() {
        speechService?.stop()
        speechService?.shutdown()
        speechService = null
    }

    override fun onDestroy() {
        super.onDestroy()
        stopVoskRecognition()
        model?.close()
    }
}

@Composable
fun MainCompose(
    viewModel: MainViewModel,
    clipboard: ClipboardManager,
    dm: DownloadManager,
    models: List<Downloadable>,
    startRecognition: ((String) -> Unit, (String) -> Unit) -> Unit,
    stopRecognition: () -> Unit
) {
    // 1. Column(arg1, arg2, arg3:lambda){}
    // 1.1 可以写成Column(arg1, arg2){arg3}，只不过这个arg3其实可以写很多个按钮实例，最后被打包在一起作为一个arg3整体
    Column(
        horizontalAlignment = Alignment.CenterHorizontally, // 整体水平居中
        modifier = Modifier.fillMaxSize() // 占满整个屏幕
    ) {
        // 1. 顶部AppBar
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.background)
                .statusBarsPadding()
                .padding(vertical = 16.dp, horizontal = 16.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(13.dp)
                    .background(MaterialTheme.colorScheme.primary, shape = RoundedCornerShape(6.5.dp))
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "LlamaBot",
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.titleLarge
            )
        }

        // 2. 对话内容
        val scrollState = rememberLazyListState()
        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            // 1.1 LazyColumn是真正用于显示内容
            LazyColumn(
                state = scrollState,
                modifier = Modifier.fillMaxSize()
            ) {
                items(viewModel.messages.size) { index ->
                    val message = viewModel.messages[index]
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 4.dp),
                        horizontalArrangement = if (index % 2 == 0) Arrangement.End else Arrangement.Start
                    ) {
                        // 这里设置文字颜色
                        // 为LocalContentColor.current，实际上就是colorScheme.onTertiary，设置成黑色
                        // 然后文字的背景设置为colorScheme.onTertiary，加上0.1透明度
                        Text(
                            text = message,
                            style = MaterialTheme.typography.bodyLarge.copy(color = LocalContentColor.current),
                            modifier = Modifier
                                .background(MaterialTheme.colorScheme.onTertiary.copy(alpha = 0.1f), shape = RoundedCornerShape(8.dp))
                                .padding(8.dp)
                        )
                    }
                }
            }

            // 1.2 LaunchedEffect用于计算出对话总长度来自动给滚动到底部
            LaunchedEffect(viewModel.messages) {
                snapshotFlow { viewModel.messages.lastOrNull() }
                    .collect { _ ->
                        if (viewModel.messages.isNotEmpty()) {
                            scrollState.animateScrollToItem(viewModel.messages.size - 1)
                        }
                    }
            }
        }

        // 3. 用户输入
        // 3.1 语音识别 - 使用Vosk离线识别
        var isRecording by remember { mutableStateOf(false) }

        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth(0.9f)
        ){
            // 3.1 获取用户的输入，并且赋值给viewModel的message属性
            OutlinedTextField(
                value = viewModel.message,
                onValueChange = { viewModel.updateMessage(it) },
                label = { Text("Message") },

                modifier = Modifier.fillMaxWidth(0.65f), // 可选，限制宽度并居中
                shape = RoundedCornerShape(24.dp) // 设置圆角
            )
            Spacer(modifier = Modifier.width(8.dp))

            Box(
                modifier = Modifier
                    .size(48.dp)
                    .background(
                        color = if (isRecording) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.secondary,
                        shape = RoundedCornerShape(50)
                    )
                    .pointerInput(Unit) {
                        detectTapGestures(
                            onPress = {
                                // 1. 开始录音 - 使用Vosk
                                isRecording = true
                                // viewModel.log("开始使用Vosk离线语音识别...")

                                startRecognition(
                                    // onResult - 最终结果
                                    { text ->
                                        viewModel.updateMessage(text)
                                        // viewModel.log("识别完成: $text")
                                    },
                                    // onPartialResult - 部分结果
                                    { partial ->
                                        viewModel.updateMessage(partial)
                                    }
                                )

                                tryAwaitRelease()
                                // 结束录音
                                isRecording = false
                                stopRecognition()
                                // viewModel.log("停止语音识别")
                            }
                        )
                    },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Mic,
                    contentDescription = "Voice Input",
                    tint = MaterialTheme.colorScheme.onSecondary
                )
            }
            Spacer(modifier = Modifier.width(8.dp))

            IconButton(
                onClick = { viewModel.send() },
                modifier = Modifier
                    .background(
                        color = MaterialTheme.colorScheme.primary,
                        shape = RoundedCornerShape(50)
                    )
                    .size(48.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.ArrowUpward,
                    contentDescription = "Send",
                    tint = MaterialTheme.colorScheme.onPrimary
                )
            }
        }

        // 3. 下载按钮
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            for (model in models) {
                Downloadable.Button(viewModel, dm, model)
            }
        }
    }
}
