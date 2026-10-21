// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.main

import android.Manifest
import org.json.JSONObject
import java.io.File

import android.os.Bundle
import android.util.Log
import android.util.TypedValue
import android.view.Menu
import android.view.MenuInflater
import android.view.MenuItem
import android.view.View
import android.view.ViewTreeObserver
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SearchView
import androidx.appcompat.widget.Toolbar
import androidx.core.view.GravityCompat
import androidx.core.view.MenuProvider
import androidx.drawerlayout.widget.DrawerLayout
import androidx.fragment.app.Fragment
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatRouter
import com.alibaba.mnnllm.android.history.ChatHistoryFragment
import com.alibaba.mnnllm.android.mainsettings.MainSettingsActivity
import com.alibaba.mnnllm.android.modelmarket.ModelMarketFragment
import com.alibaba.mnnllm.android.update.UpdateChecker
import com.alibaba.mnnllm.android.utils.CrashUtil
import com.alibaba.mnnllm.android.utils.RouterUtils.startActivity
import com.alibaba.mnnllm.android.utils.Searchable
import com.alibaba.mnnllm.android.widgets.BottomTabBar
import com.alibaba.mnnllm.android.widgets.ModelSwitcherView
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.google.android.material.appbar.AppBarLayout
import com.google.android.material.appbar.MaterialToolbar
import com.alibaba.mnnllm.android.chat.SelectSourceFragment
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.speech.tts.TextToSpeech
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.privacy.PrivacyPolicyManager
import com.alibaba.mnnllm.android.privacy.PrivacyPolicyDialogFragment
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService
import kotlin.random.Random

class MainActivity : AppCompatActivity(), MainFragmentManager.FragmentLifecycleListener {
    private lateinit var drawerLayout: DrawerLayout
    private var toggle: ActionBarDrawerToggle? = null
    private lateinit var appBarLayout: AppBarLayout
    private lateinit var materialToolbar: MaterialToolbar
    private lateinit var mainTitleSwitcher: ModelSwitcherView
    private var toolbarHeightPx: Int = 0
    private var offsetChangedListener: AppBarLayout.OnOffsetChangedListener? = null
    private var chatHistoryFragment: ChatHistoryFragment? = null
    private var updateChecker: UpdateChecker? = null
    private lateinit var expandableFabLayout: View
    
    // Add field to track current search view
    private var currentSearchView: SearchView? = null

    private lateinit var bottomNav: BottomTabBar
    private lateinit var mainFragmentManager: MainFragmentManager

    private val currentFragment: Fragment?
        get() {
            return mainFragmentManager.activeFragment
        }




    private lateinit var recognizer: Recognizer
    private lateinit var speechService: SpeechService
    private lateinit var tts: TextToSpeech
    private lateinit var tvAsrText: TextView
    private fun startListening() {
        Log.d("hygebra", "startListening")
        speechService = SpeechService(recognizer, 16000.0f)
        Log.d("hygebra", "startListening SpeechService ended")
        speechService.startListening(object : RecognitionListener {

            override fun onPartialResult(hypothesis: String?) {
                val partial = JSONObject(hypothesis ?: "{}").optString("partial")
                runOnUiThread {
                    tvAsrText.text = partial
                    Log.d("hygebra", partial)
                }
            }

            override fun onResult(hypothesis: String) {
                val partial = JSONObject(hypothesis).optString("partial")
                runOnUiThread {
                    tvAsrText.text = partial
                    Log.d("hygebra", partial)
                }
            }

            override fun onFinalResult(hypothesis: String) {
                val text = JSONObject(hypothesis).optString("text")
                runOnUiThread {
                    tvAsrText.text = text
                    Log.d("hygebra", text)
                }
            }

            override fun onError(e: Exception) {
                e.printStackTrace()
            }

            override fun onTimeout() {}
        })
    }


    fun startAsr() {
        Log.d("hygebra", "startAsr")
        StorageService.unpack(
            this,
            "model-cn",   // assets 下的目录名
            "model-cn",      // 解压到 app 私有目录
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
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                1001
            )
        } else {
            try{
                startAsr()
            }catch (e:Exception){
                e.printStackTrace()
            }
        }
    }


    private var quant_kv: String? = ""
    private var run_device: String? = "cpu"

    private fun changeTitle(
    ) {
        var text: String = "kv量化"
        if (quant_kv=="") {
            text = text + "无"
        } else {
            text = text + quant_kv
        }
        text = text + "+"
        text = text + run_device
        mainTitleSwitcher.text=text
        if (run_device == "cpu" && quant_kv == "") {
            mainTitleSwitcher.setBackgroundColor(Color.rgb(255,255,255))
        } else {
            val color = Color.rgb(
                Random.nextInt(128)+128,
                Random.nextInt(128)+128,
                Random.nextInt(128)+128
            )
            materialToolbar.setBackgroundColor(color)
        }
    }

    fun updateSingleConfigJson(
        configFile: File,
        key: String,
        value: String
    ) {
        // 读取原始 JSON
        val jsonText = configFile.readText()
        val json = JSONObject(jsonText)
        Log.e("hygebra", "updateSingleConfigJson: $json")

        // 根据 value 内容推断类型
        val parsedValue: Any = when {
            value.equals("true", true) -> true
            value.equals("false", true) -> false
            value.toIntOrNull() != null -> value.toInt()
            value.toFloatOrNull() != null -> value.toFloat()
            else -> value
        }

        if (parsedValue == "") {
            json.remove(key)
        } else {
            // 修改 / 新增字段
            json.put(key, parsedValue)
        }

        // 写回（保持 pretty format）
        configFile.writeText(json.toString(4))
    }

    fun updateConfigJson(
        key: String,
        value: String
    ) {
        try {
            val modelsDir = File("/data/local/tmp/mnn_models/")
            if (modelsDir.exists() && modelsDir.isDirectory) {
                modelsDir.listFiles()?.forEach { modelDir ->
                    Log.d(TAG, "updateConfigJson: modelDir: $modelDir")
                    if (modelDir.isDirectory && File(modelDir, "config.json").exists()) {
                        val modelPath = modelDir.absolutePath
                        val modelId = "local/${modelPath}"
                        updateSingleConfigJson(File(modelDir, "config.json"), key=key, value=value)
                        Log.d(TAG, "updateConfigJson: modelId: $modelId")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("ModelUtils", "Failed to load models from /data/local/tmp/mnn_models/", e)
        }
    }


    private val menuProvider: MenuProvider = object : MenuProvider {
        override fun onCreateMenu(menu: Menu, menuInflater: MenuInflater) {
            menuInflater.inflate(R.menu.menu_main, menu)
            setupSearchView(menu)
            setupOtherMenuItems(menu)
        }

        override fun onMenuItemSelected(menuItem: MenuItem): Boolean {
            return true
        }


        override fun onPrepareMenu(menu: Menu) {
            Log.d(TAG, "onPrepareMenu")
            super.onPrepareMenu(menu)
            val searchItem = menu.findItem(R.id.action_search)
            val reportCrashMenu = menu.findItem(R.id.action_report_crash)
            reportCrashMenu.isVisible = CrashUtil.hasCrash()

            // Show/hide search based on current fragment
            searchItem.isVisible = when (bottomNav.getSelectedTab()) {
                BottomTabBar.Tab.LOCAL_MODELS, BottomTabBar.Tab.MODEL_MARKET -> true
                else -> false
            }
        }
    }

    private fun setupSearchView(menu: Menu) {
        val searchItem = menu.findItem(R.id.action_search)
        val searchView = searchItem.actionView as SearchView?
        if (searchView != null) {
            currentSearchView = searchView
            searchView.setOnQueryTextListener(object : SearchView.OnQueryTextListener {
                override fun onQueryTextSubmit(query: String): Boolean {
                    handleSearch(query)
                    return false
                }

                override fun onQueryTextChange(query: String): Boolean {
                    handleSearch(query)
                    return true
                }
            })
            searchItem.setOnActionExpandListener(object : MenuItem.OnActionExpandListener {
                override fun onMenuItemActionExpand(item: MenuItem): Boolean {
                    Log.d(TAG, "SearchView expanded")
                    return true
                }

                override fun onMenuItemActionCollapse(item: MenuItem): Boolean {
                    Log.d(TAG, "SearchView collapsed")
                    handleSearchCleared()
                    return true
                }
            })
        }
    }

    private fun setupOtherMenuItems(menu: Menu) {
        updateConfigJson("quant_kv", "")
        updateConfigJson("run_device", "cpu")
        val issueMenu = menu.findItem(R.id.action_kv_q8)
        issueMenu.setOnMenuItemClickListener {
            setKV_q8(null)
            true
        }

        val noneMenu = menu.findItem(R.id.action_kv_none)
        noneMenu.setOnMenuItemClickListener {
            setKV_none(null)
            true
        }
        val noneMenu1 = menu.findItem(R.id.action_device_cpu)
        noneMenu1.setOnMenuItemClickListener {
            setDevice_cpu(null)
            true
        }
        val noneMenu2 = menu.findItem(R.id.action_device_opencl)
        noneMenu2.setOnMenuItemClickListener {
            setDevice_opencl(null)
            true
        }

        val settingsMenu = menu.findItem(R.id.action_settings)
        settingsMenu.setOnMenuItemClickListener {
            startActivity(this@MainActivity, MainSettingsActivity::class.java)
            true
        }

        val starGithub = menu.findItem(R.id.action_kv_q4)
        starGithub.setOnMenuItemClickListener {
            setKV_fp8(null)
            true
        }

        val reportCrashMenu = menu.findItem(R.id.action_add_models)
        reportCrashMenu.setOnMenuItemClickListener {
            set_kv(null)
            true
        }
    }

    private fun handleSearch(query: String) {
        val searchableFragment = currentFragment as? Searchable
        searchableFragment?.onSearchQuery(query)
    }

    private fun handleSearchCleared() {
        val searchableFragment = currentFragment as? Searchable
        searchableFragment?.onSearchCleared()
    }

    /**
     * Set the SearchView query and expand it if needed
     */
    fun setSearchQuery(query: String) {
        if (query.isEmpty()) return
        
        val menu = materialToolbar.menu
        val searchItem = menu?.findItem(R.id.action_search)
        
        if (searchItem != null && searchItem.isVisible) {
            try {
                // Expand the search view first
                searchItem.expandActionView()
                
                // Set the query after expansion
                currentSearchView?.let { searchView ->
                    searchView.setQuery(query, false)
                    searchView.clearFocus() // Prevent automatic keyboard popup
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to set search query: $query", e)
            }
        }
    }
    
    /**
     * Get the current search query
     */
    fun getCurrentSearchQuery(): String {
        return currentSearchView?.query?.toString() ?: ""
    }
    
    /**
     * Clear the search query and collapse the SearchView
     */
    fun clearSearch() {
        val menu = materialToolbar.menu
        val searchItem = menu?.findItem(R.id.action_search)
        searchItem?.collapseActionView()
    }

    private fun setupAppBar() {
        appBarLayout = findViewById(R.id.app_bar)
        materialToolbar = findViewById(R.id.toolbar)
        mainTitleSwitcher = findViewById(R.id.main_title_switcher)

        // Initially hide the dropdown arrow and make it non-clickable
        updateMainTitleSwitcherMode(false)

        toolbarHeightPx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            48f, // Toolbar height in DP from your XML
            resources.displayMetrics
        ).toInt()

        materialToolbar.viewTreeObserver.addOnGlobalLayoutListener(object : ViewTreeObserver.OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                materialToolbar.viewTreeObserver.removeOnGlobalLayoutListener(this)
                val measuredHeight = materialToolbar.height
                if (measuredHeight > 0) {
                    toolbarHeightPx = measuredHeight
                }
            }
        })

        offsetChangedListener = AppBarLayout.OnOffsetChangedListener { appBarLayout, verticalOffset ->
            if (toolbarHeightPx <= 0) {
                val currentToolbarHeight = materialToolbar.height
                if (currentToolbarHeight > 0) {
                    toolbarHeightPx = currentToolbarHeight
                } else {
                    toolbarHeightPx = TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, 48f, resources.displayMetrics).toInt()
                    if (toolbarHeightPx == 0) return@OnOffsetChangedListener // Still zero, cannot proceed
                }
            }
            val absVerticalOffset = Math.abs(verticalOffset)
            var alpha = 1.0f - (absVerticalOffset.toFloat() / toolbarHeightPx.toFloat())
            alpha = alpha.coerceIn(0.0f, 1.0f)
            materialToolbar.alpha = alpha
        }
        //appBarLayout.addOnOffsetChangedListener(offsetChangedListener)
    }

    /**
     * Update the mode of the main title switcher
     * @param isSourceSwitcherMode Whether it is in source switcher mode (shows dropdown arrow and is clickable)
     */
    private fun updateMainTitleSwitcherMode(isSourceSwitcherMode: Boolean) {
        val dropdownArrow = mainTitleSwitcher.findViewById<View>(R.id.iv_dropdown_arrow)
        if (isSourceSwitcherMode) {
            // Source switcher mode: show dropdown arrow, clickable
            dropdownArrow?.visibility = View.VISIBLE
            mainTitleSwitcher.isClickable = true
            mainTitleSwitcher.isFocusable = true
            mainTitleSwitcher.setOnClickListener {
                // Show source selection dialog
                showSourceSelectionDialog()
            }
        } else {
            // Title display mode: hide dropdown arrow, not clickable
            dropdownArrow?.visibility = View.GONE
            mainTitleSwitcher.isClickable = false
            mainTitleSwitcher.isFocusable = false
            mainTitleSwitcher.setOnClickListener(null)
        }
    }

    /**
     * Show source selection dialog
     */
    private fun showSourceSelectionDialog() {
        val availableSources = ModelSources.sourceList
        val displayNames = ModelSources.sourceDisPlayList
        val currentProvider = MainSettings.getDownloadProviderString(this)
        
        // Use SelectSourceFragment from ModelMarketFragment
        val fragment = SelectSourceFragment.newInstance(availableSources, displayNames, currentProvider)
        fragment.setOnSourceSelectedListener { selectedSource ->
            MainSettings.setDownloadProvider(this, selectedSource)
            // Set title to display name
            val idx = ModelSources.sourceList.indexOf(selectedSource)
            val displayName = if (idx != -1) getString(ModelSources.sourceDisPlayList[idx]) else selectedSource
            mainTitleSwitcher.text = displayName
            // Notify ModelMarketFragment to update
            if (currentFragment is ModelMarketFragment) {
                (currentFragment as ModelMarketFragment).onSourceChanged()
            }
        }
        fragment.show(supportFragmentManager, "SourceSelectionDialog")
    }

    private fun updateExpandableFabLayout(newTab: BottomTabBar.Tab) {
        expandableFabLayout.visibility = if (newTab == BottomTabBar.Tab.LOCAL_MODELS) {
            View.VISIBLE
        } else {
            View.GONE
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        mainFragmentManager.onSaveInstanceState(outState)
    }

    override fun onTabChanged(newTab: BottomTabBar.Tab) {
        Log.d(TAG, "Tab changed to $newTab, updating UI accordingly.")

        when (newTab) {
            BottomTabBar.Tab.LOCAL_MODELS -> {
                updateMainTitleSwitcherMode(false)
                mainTitleSwitcher.text = getString(R.string.nav_name_chats)
            }
            BottomTabBar.Tab.MODEL_MARKET -> {
                updateMainTitleSwitcherMode(true)
                val currentProvider = MainSettings.getDownloadProviderString(this)
                val idx = ModelSources.sourceList.indexOf(currentProvider)
                val displayName = if (idx != -1) getString(ModelSources.sourceDisPlayList[idx]) else currentProvider
                mainTitleSwitcher.text = displayName
            }
            BottomTabBar.Tab.BENCHMARK -> {
                updateMainTitleSwitcherMode(false)
                mainTitleSwitcher.text = getString(R.string.benchmark)
            }
        }
        updateExpandableFabLayout(newTab)
        invalidateOptionsMenu()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Check privacy policy agreement first
        checkPrivacyPolicyAgreement()
        
        setContentView(R.layout.activity_main)
        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setupAppBar()
        bottomNav = findViewById(R.id.bottom_navigation)
        drawerLayout = findViewById(R.id.drawer_layout)
        expandableFabLayout = findViewById(R.id.expandable_fab_layout)
        updateChecker = UpdateChecker(this)
        updateChecker!!.checkForUpdates(this, false)
        mainFragmentManager = MainFragmentManager(this, R.id.main_fragment_container, bottomNav, this)
        mainFragmentManager.initialize(savedInstanceState)
        Log.d(TAG, "onCreate: Before bottomNav.select, currentFragment: ${currentFragment?.javaClass?.simpleName}")

        //hygebra
        setKV_none(null)
        setDevice_cpu(null)
        changeTitle()

        setSupportActionBar(toolbar)

        supportActionBar?.apply {
            setDisplayHomeAsUpEnabled(true)   // 显示返回箭头
            setDisplayShowHomeEnabled(true)
        }
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (drawerLayout.isDrawerOpen(GravityCompat.START)) {
                    drawerLayout.closeDrawer(GravityCompat.START)
                } else {
                    finish()
                }
            }
        })
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.setDisplayShowTitleEnabled(false)  // Disable default title display
        
        // Handle intent extras for navigation from notification
        handleIntentExtras(intent)
    }
    
    private fun handleIntentExtras(intent: Intent?) {
        intent?.let {
            val selectTab = it.getStringExtra(EXTRA_SELECT_TAB)
            if (selectTab == TAB_MODEL_MARKET) {
                // Post to ensure the UI is ready
                bottomNav.post {
                    bottomNav.select(BottomTabBar.Tab.MODEL_MARKET)
                }
            }
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (toggle!!.onOptionsItemSelected(item)) {
            return true
        }
        return super.onOptionsItemSelected(item)
    }
    
    fun runModel(destModelDir: String?, modelIdParam: String?, sessionId: String?) {
        ChatRouter.startRun(this, modelIdParam!!, destModelDir, sessionId)
        drawerLayout.close()
    }

    fun setKV_fp8(view: View?) {
        quant_kv = "Vfp8"
        updateConfigJson("quant_qkv", "2")
        changeTitle()
    }

    fun setKV_q8(view: View?) {
        quant_kv = "K非对称8b"
        updateConfigJson("quant_qkv", "1")
        changeTitle()
    }
    fun setKV_none(view: View?) {
        quant_kv = ""
        updateConfigJson("quant_qkv", "0")
        changeTitle()
    }

    fun setDevice_cpu(view: View?) {
        run_device = "cpu"
        updateConfigJson("backend_type", "cpu")
        changeTitle()
    }
    fun setDevice_opencl(view: View?) {
        run_device = "opencl"
        updateConfigJson("backend_type", "opencl")
        changeTitle()
    }


    fun set_kv(view: View?) {

        quant_kv = "K非对称8b+Vfp8"
        updateConfigJson("quant_qkv", "3")
        changeTitle()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == ModelDownloadManager.REQUEST_CODE_POST_NOTIFICATIONS) {
            ModelDownloadManager.getInstance(this).tryStartForegroundService()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    fun onAddModelButtonClick(view: View) {
        bottomNav.select(BottomTabBar.Tab.MODEL_MARKET)
    }
    
    /**
     * Check if user has agreed to privacy policy
     * If not, show privacy policy dialog
     */
    private fun checkPrivacyPolicyAgreement() {
        if (!ENABLE_PRIVACY_POLICY_CHECK) {
            return
        }
        
        val privacyManager = PrivacyPolicyManager.getInstance(this)
        
        if (!privacyManager.hasUserAgreed()) {
            showPrivacyPolicyDialog()
        }
    }
    
    /**
     * Show privacy policy dialog
     */
    private fun showPrivacyPolicyDialog() {
        val dialog = PrivacyPolicyDialogFragment.newInstance(
            onAgree = {
                // User agreed to privacy policy
                val privacyManager = PrivacyPolicyManager.getInstance(this)
                privacyManager.setUserAgreed(true)
                Log.d(TAG, "User agreed to privacy policy")
            },
            onDisagree = {
                // User disagreed to privacy policy
                Toast.makeText(this, getString(R.string.privacy_policy_exit_message), Toast.LENGTH_LONG).show()
                Log.d(TAG, "User disagreed to privacy policy")
                // Exit the application
                finishAffinity()
            }
        )
        
        dialog.show(supportFragmentManager, PrivacyPolicyDialogFragment.TAG)
    }

    companion object {
        const val TAG: String = "MainActivity"
        const val EXTRA_SELECT_TAB = "com.alibaba.mnnllm.android.select_tab"
        const val TAB_MODEL_MARKET = "model_market"
        
        // Control whether to show privacy policy agreement dialog
        const val ENABLE_PRIVACY_POLICY_CHECK = false
    }
}