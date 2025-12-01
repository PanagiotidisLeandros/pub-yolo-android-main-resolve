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

class MainActivity : AppCompatActivity(){

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
    }

    override fun onDestroy() {
        super.onDestroy()
        //speechRecognizer.destroy()
        textToSpeech.shutdown()
    }

    private fun showFragment(fragment: Fragment) {
        supportFragmentManager
            .beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .commit()
    }
}