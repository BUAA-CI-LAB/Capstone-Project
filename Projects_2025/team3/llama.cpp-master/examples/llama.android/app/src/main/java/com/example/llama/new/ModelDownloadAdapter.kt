package com.example.llama.new

import android.view.View
import android.view.ViewGroup
import android.view.LayoutInflater
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.llama.R

data class ModelItem(
    val name: String,
    val url: String,
    var status: String = "未下载"
)

class ModelDownloadAdapter(
    private val models: MutableList<ModelItem>,
    private val onDownloadClick: (ModelItem, Int) -> Unit
) : RecyclerView.Adapter<ModelDownloadAdapter.ModelViewHolder1>() {

    inner class ModelViewHolder1(view: View) : RecyclerView.ViewHolder(view) {
        val modelName: TextView = view.findViewById(R.id.model_name)
        val downloadBtn: Button = view.findViewById(R.id.download_btn)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelViewHolder1 {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_model, parent, false)
        return ModelViewHolder1(view)
    }

    override fun onBindViewHolder(holder: ModelViewHolder1, position: Int) {
        val model = models[position]
        holder.modelName.text = "${model.name} (${model.status})"
        holder.downloadBtn.text = if (model.status == "已下载") "打开" else "下载"

        holder.downloadBtn.setOnClickListener {
            onDownloadClick(model, position)
        }
    }

    override fun getItemCount(): Int = models.size

    fun updateModel(position: Int, model: ModelItem) {
        models[position] = model
        notifyItemChanged(position)
    }
}
