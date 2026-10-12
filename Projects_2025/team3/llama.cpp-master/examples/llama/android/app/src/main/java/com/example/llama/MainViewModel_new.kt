package com.example.llama

import android.llama.cpp.LLamaAndroid
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow

class MainViewModel_new : ViewModel() {
    var currentModel = MutableStateFlow<String?>(null)
}
