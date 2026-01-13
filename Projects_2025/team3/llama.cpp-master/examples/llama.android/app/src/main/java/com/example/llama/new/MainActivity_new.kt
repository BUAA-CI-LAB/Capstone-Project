package com.example.llama.new

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.Menu
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.lifecycle.lifecycleScope
import com.example.llama.R
import com.google.android.material.navigation.NavigationView
import kotlinx.coroutines.launch
import java.io.File
import kotlin.toString

class MainActivity_new : AppCompatActivity() {

    lateinit var drawerLayout: DrawerLayout
    lateinit var navigationView : NavigationView

    private var currentBottomSheet: ModelSelectBottomSheet? = null
    private lateinit var chatFragment: ChatFragment

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.test_activity_main)

        drawerLayout = findViewById(R.id.main)
        navigationView = findViewById(R.id.nav_view)

        if (savedInstanceState == null) {
            showChat()
        }

//        HistoryManager.clearHistory(this)
        refreshHistoryMenu()

        navigationView.setNavigationItemSelectedListener { item ->
            when (item.itemId) {
                R.id.menu_new_chat -> {
                    openNewConversation()
                    refreshHistoryMenu()
                }
                R.id.menu_clear_history -> {
                    HistoryManager.clearHistory(this)
                    refreshHistoryMenu()
                }
                else -> {
                    val conversationId = item.titleCondensed.toString()
                    openConversation(conversationId)
                    refreshHistoryMenu()
                }
            }
            drawerLayout.closeDrawers()
            true
        }
    }

    private var currentConversationId: String? = null

    fun openNewConversation() {
        currentConversationId = null

        chatFragment.openNewConversation()
    }

    fun openConversation(id: String) {
        currentConversationId = id
        val conversations = HistoryManager.loadConversations(this)
        val conv = conversations.find { it.id == id } ?: return

        chatFragment.loadConversation(conv)
    }

    fun refreshHistoryMenu() {
        val menu = navigationView.menu

        // 先清空旧的历史项（保留固定菜单）
        menu.removeGroup(R.id.group_history)

        // 读取历史会话
        val conversations = HistoryManager.loadConversations(this)

        conversations.asReversed().forEach { conv ->
            Log.d("refresh", conv.title)
            val item = menu.add(
                R.id.group_history,         // groupId
                View.generateViewId(),         // itemId（唯一）
                Menu.NONE,
                conv.title           // 显示标题
            )
            conv.title.forEach { c -> Log.d("charCode", "${c.code}") }
            item.titleCondensed = conv.id
            item.isCheckable = true

            if (conv.id == currentConversationId) {
                item.isChecked = true
            }
        }
    }

    fun showDeleteDialog(conversation: Conversation) {
        AlertDialog.Builder(this)
            .setTitle("删除对话")
            .setMessage("确定要删除「${conversation.title}」吗？")
            .setPositiveButton("删除") { _, _ ->
                HistoryManager.deleteConversation(this, conversation.id)
                refreshHistoryMenu()
                chatFragment.onConversationDeleted(conversation.id)
                if (currentConversationId == conversation.id) {
                    currentConversationId = null
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    fun showChat() {
        chatFragment = ChatFragment()
        supportFragmentManager.beginTransaction()
            .replace(R.id.container, chatFragment)
            .commit()
    }

    fun showModelDownload() {
        supportFragmentManager.beginTransaction()
            .replace(R.id.container, ModelDownloadFragment())
            .addToBackStack("model_download")
            .commit()
    }

    fun openDrawer() {
        drawerLayout.openDrawer(GravityCompat.START)
    }

    private fun getFileName(uri: Uri): String {
        var name = "model.gguf"
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (cursor.moveToFirst() && nameIndex >= 0) {
                name = cursor.getString(nameIndex)
            }
        }
        return name
    }

    private fun copyUriToFile(uri: Uri): File {
        val fileName = getFileName(uri) // 可以从 uri 获取真实名字
        val destFile = File(filesDir, fileName)
        contentResolver.openInputStream(uri)?.use { input ->
            destFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return destFile
    }

    private val openFileLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val uri: Uri? = result.data?.data
            if (uri != null) {
                val file = copyUriToFile(uri)
                val path = file.absolutePath
                changeModel(path)
            }
        }
    }

    fun changeModel(path: String) {
        val name = File(path).name

        // 回传给 BottomSheet
        currentBottomSheet?.onLocalModelPicked(path)
        Log.d("MainActivity", "选择文件: $path")
        Toast.makeText(this, "选择文件: $path", Toast.LENGTH_SHORT).show()

        lifecycleScope.launch {
            try {
                if (chatFragment.engine.isLoaded()) {
                    chatFragment.engine.cleanUp()
                }
                chatFragment.engine.loadModel(path)
                chatFragment.changeModel(name)
            } catch (exc: IllegalStateException) {
                Log.e("TAG", "load() failed", exc)
            }
        }
    }

    fun openFileSelector(sheet: ModelSelectBottomSheet) {
        currentBottomSheet = sheet

        // 打开文件管理器选择任意文件
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*" // 或者 "application/octet-stream" 仅允许二进制文件
        }
        openFileLauncher.launch(intent)
    }
}
