package org.tensorflow.lite.examples.objectdetection

import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import org.tensorflow.lite.examples.objectdetection.fragments.CameraFragment
import org.tensorflow.lite.examples.objectdetection.fragments.PermissionsFragment

class MainActivity : AppCompatActivity() {

    private lateinit var speechRecognizer: SpeechRecognizer
    private var isListening = false
    private lateinit var textToSpeech: TextToSpeech

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize TextToSpeech
        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.speak("Application is running", TextToSpeech.QUEUE_FLUSH, null, null)
            } else {
                Toast.makeText(this, "Text-to-Speech failed to initialize", Toast.LENGTH_SHORT).show()
            }
        }

        if (PermissionsFragment.hasPermissions(this)) {
            showFragment(CameraFragment())
        } else {
            showFragment(PermissionsFragment())
        }
        setupSpeechRecognizer()
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
        textToSpeech.shutdown()
    }

    private fun showFragment(fragment: Fragment) {
        supportFragmentManager
            .beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .commit()
    }

    private fun startListening() {
        if (!isListening) {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }
            speechRecognizer.startListening(intent)
            isListening = true
        }
    }

    private fun setupSpeechRecognizer() {
        if (SpeechRecognizer.isRecognitionAvailable(this)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
            speechRecognizer.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onError(error: Int) {
                    isListening = false
                    startListening()
                }
                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    matches?.let {
                        when {
                            it.contains("show options") -> {
                                val fragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
                                if (fragment is CameraFragment) {
                                    fragment.showSettingsPanel()
                                } else {
                                    Toast.makeText(this@MainActivity, "CameraFragment not found", Toast.LENGTH_SHORT).show()
                                }
                            }
                            it.contains("hide options") -> {
                                val fragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
                                if (fragment is CameraFragment) {
                                    fragment.hideSettingsPanel()
                                }
                            }
                        }
                    }
                    isListening = false
                    startListening()
                }
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            startListening()
        } else {
            Log.w("MainActivity", "Speech recognition not available")
        }
    }
}