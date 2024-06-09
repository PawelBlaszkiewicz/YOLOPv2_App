package com.example.yolopv2app

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Switch
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.YOLOPv2App.R
import java.io.InputStream


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var inputImage: ImageView
    private lateinit var outputImage: ImageView
    private lateinit var videoSwitch: Switch
    private lateinit var cameraSwitch: Switch
    private lateinit var objectDetectionButton: Button
    private var imageid = 1;
    private lateinit var classes:List<String>

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImage = findViewById(R.id.imageView1)
        outputImage = findViewById(R.id.imageView2)
        videoSwitch = findViewById(R.id.switch1)
        videoSwitch.setOnCheckedChangeListener { _, isChecked ->
            val message = if (isChecked) "Video Mode:ON" else "Video Mode:OFF"
            Toast.makeText(
                this@MainActivity, message,
                Toast.LENGTH_SHORT
            ).show()
            val intent = Intent(this, VideoActivity::class.java)
            startActivity(intent)
        }
        cameraSwitch = findViewById(R.id.switch3)
        cameraSwitch.setOnCheckedChangeListener { _, isChecked ->
            val message = if (isChecked) "Camera Mode:ON" else "Camera Mode:OFF"
            Toast.makeText(
                this@MainActivity, message,
                Toast.LENGTH_SHORT
            ).show()
            val intent = Intent(this, CameraActivity::class.java)
            startActivity(intent)
        }
        objectDetectionButton = findViewById(R.id.object_detection_button)
        inputImage.setImageBitmap(
            BitmapFactory.decodeStream(assets.open("test_object_detection_0.jpg"))
        );
        imageid = 0
        classes = readClasses();
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        objectDetectionButton.setOnClickListener {
            try {
                performObjectDetection(ortSession)
                Toast.makeText(baseContext, "YOLOPv2 detection performed!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when perform ObjectDetection", e)
                Toast.makeText(baseContext, "Failed to perform ObjectDetection", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    private fun updateUI(result: Result) {
        val mutableBitmap: Bitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.color = Color.WHITE

        paint.textSize = 20f

        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)

        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, paint)
        var boxit = result.outputBox.iterator()
        while(boxit.hasNext()) {
            var box_info = boxit.next()
            canvas.drawText("%s:%.2f".format(classes[box_info[5].toInt()],box_info[4]),
                box_info[0]-box_info[2]/2, box_info[1]-box_info[3]/2, paint)
        }
        outputImage.setImageBitmap(mutableBitmap)
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.yolopv2_192x320
        Log.d("YOLOPV2", "Yolopv2 192x320 is loaded!")
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    private var imageCounter = 0

    private fun readInputImage(): InputStream {
        val imageName = "test_object_detection_${imageCounter}.jpg"
        imageCounter = (imageCounter + 1) % 3
        return assets.open(imageName)
    }

    private fun performObjectDetection(ortSession: OrtSession) {
        val objDetector = ObjectDetector()
        val imagestream = readInputImage()
        val imageBitmap = BitmapFactory.decodeStream(imagestream)
        inputImage.setImageBitmap(imageBitmap);
        val result = objDetector.detect(imageBitmap, ortEnv, ortSession, classes)
        imagestream.reset()
        updateUI(result);
    }

    companion object {
        const val TAG = "YOLOPv2Detection"
    }
}