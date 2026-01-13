package com.example.llama.new

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView



/**
 * Adapter 用于显示模型列表（默认模型 + 本地模型）
 */
class ModelAdapter(
    private val models: MutableList<LocalModel>,
    private val onClick: (LocalModel) -> Unit
) : RecyclerView.Adapter<ModelAdapter.ModelViewHolder>() {

    data class LocalModel(
        val name: String, // 显示名称
        val path: String  // 文件路径
    )

    inner class ModelViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val text: TextView = view.findViewById(android.R.id.text1)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(android.R.layout.simple_list_item_1, parent, false)
        return ModelViewHolder(view)
    }

    override fun onBindViewHolder(holder: ModelViewHolder, position: Int) {
        val model = models[position]
        holder.text.text = model.name
        holder.itemView.setOnClickListener { onClick(model) }
    }

    override fun getItemCount(): Int = models.size

    fun addModel(model: LocalModel) {
        models.add(model)
        notifyItemInserted(models.size - 1)
    }
}
