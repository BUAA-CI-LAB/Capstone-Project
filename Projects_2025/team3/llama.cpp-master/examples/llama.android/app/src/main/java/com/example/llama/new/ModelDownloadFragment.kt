package com.example.llama.new

import android.os.Bundle
import android.view.View
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.RecyclerView
import com.example.llama.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import android.widget.Toast
import kotlinx.coroutines.Dispatchers
import androidx.recyclerview.widget.LinearLayoutManager
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.cancel
import java.io.File
import java.io.FileOutputStream
import java.net.URL

class ModelDownloadFragment : Fragment(R.layout.fragment_model_download) {

    private lateinit var recycler: RecyclerView
    private lateinit var adapter: ModelDownloadAdapter
    private val models = mutableListOf(
        ModelItem("qwen2-1.5b-instruct-q8_0", "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/blob/main/qwen2-1_5b-instruct-q8_0.gguf")
    )

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        recycler = view.findViewById(R.id.models_recycler)
        recycler.layoutManager = LinearLayoutManager(requireContext())

        adapter = ModelDownloadAdapter(models) { model, pos ->
            if (model.status == "已下载") {
                Toast.makeText(requireContext(), "模型已下载：${model.name}", Toast.LENGTH_SHORT).show()
            } else {
                downloadModel(model, pos)
            }
        }

        recycler.adapter = adapter
    }

    private fun downloadModel(model: ModelItem, pos: Int) {
        adapter.updateModel(pos, model.copy(status = "下载中..."))

        scope.launch {
            try {
                val url = URL(model.url)
                val connection = url.openConnection()
                connection.connect()
                val totalSize = connection.contentLength

                val input = url.openStream()
                val file = File(requireContext().filesDir, model.name + ".gguf")
                val output = FileOutputStream(file)

                val buffer = ByteArray(8 * 1024)
                var bytesRead: Int
                var downloaded = 0

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    downloaded += bytesRead
                    val progress = (downloaded * 100 / totalSize.toFloat()).toInt()
                    // 可选：更新进度
                }

                output.flush()
                output.close()
                input.close()

                withContext(Dispatchers.Main) {
                    adapter.updateModel(pos, model.copy(status = "已下载"))
                    Toast.makeText(requireContext(), "${model.name} 下载完成", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    adapter.updateModel(pos, model.copy(status = "下载失败"))
                    Toast.makeText(requireContext(), "${model.name} 下载失败", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        scope.cancel()
    }
}
