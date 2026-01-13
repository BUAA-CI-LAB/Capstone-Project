package com.example.llama.new


import android.os.Bundle
import android.view.ViewGroup
import com.example.llama.R
import java.io.File
import android.content.Context
import android.view.LayoutInflater
import android.view.View
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetDialogFragment

class ModelSelectBottomSheet(
    private val onAction: (Action) -> Unit
) : BottomSheetDialogFragment() {

    enum class Action { DOWNLOAD, MODEL_SELECTED }

    private val localModels = mutableListOf<ModelAdapter.LocalModel>()
    private lateinit var adapter: ModelAdapter

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.sheet_model_select, container, false)

        localModels.addAll(loadLocalModelsFromPrefs())
        val modelList = localModels
        val recycler = view.findViewById<RecyclerView>(R.id.recycler)
        recycler.layoutManager = LinearLayoutManager(context)

        adapter = ModelAdapter(modelList) { model ->
            (requireActivity() as MainActivity_new).changeModel(model.path)
            onAction(Action.MODEL_SELECTED)
            dismiss()
        }
        recycler.adapter = adapter

        view.findViewById<View>(R.id.download).setOnClickListener {
            onAction(Action.DOWNLOAD)
            dismiss()
        }

        view.findViewById<View>(R.id.loadLocal).setOnClickListener {
            (requireActivity() as MainActivity_new).openFileSelector(this)
        }

        return view
    }

    fun onLocalModelPicked(path: String) {
        val name = File(path).name // 从路径生成显示名称
        val model = ModelAdapter.LocalModel(name, path)

        // 防止重复
        if (localModels.none { it.path == path }) {
            localModels.add(model)
            adapter.addModel(model)
            saveLocalModelToPrefs(model)
        }

        dismiss()
    }

    private fun saveLocalModelToPrefs(model: ModelAdapter.LocalModel) {
        val prefs = requireContext().getSharedPreferences("local_models", Context.MODE_PRIVATE)
        val set = prefs.getStringSet("models", mutableSetOf())?.toMutableSet() ?: mutableSetOf()
        set.add("${model.path}|${model.name}")
        prefs.edit().putStringSet("models", set).apply()
    }

    private fun loadLocalModelsFromPrefs(): List<ModelAdapter.LocalModel> {
        val prefs = requireContext().getSharedPreferences("local_models", Context.MODE_PRIVATE)
        val set = prefs.getStringSet("models", emptySet()) ?: emptySet()
        return set.mapNotNull {
            val parts = it.split("|")
            if (parts.size == 2) ModelAdapter.LocalModel(parts[1], parts[0]) else null
        }
    }
}
