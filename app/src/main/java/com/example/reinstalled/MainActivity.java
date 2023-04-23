package com.example.reinstalled;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import org.tensorflow.lite.support.image.TensorImage;
import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.reinstalled.ml.Yolov3416Fp16;

import org.tensorflow.lite.DataType;
//import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    Button selectBtn , predictionBtn, captureBtn;
    TextView result;
    Bitmap bitmap;
    ImageView imageview;
    @Nullable
    private Intent data;


    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //permission
        getPermission();

        selectBtn = findViewById(R.id.selectBtn);
        predictionBtn = findViewById(R.id.predictBtn);
        captureBtn = findViewById(R.id.captureBtn);
        imageview = findViewById(R.id.imageview);
        result = findViewById(R.id.result);
        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent , 10);


            }
        });

        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);

            }
        });

        predictionBtn.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View v) {


                try {
                    Yolov3416Fp16 model = Yolov3416Fp16.newInstance(MainActivity.this);



                    // Create an input tensor buffer with the expected shape
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 416, 416, 3}, DataType.FLOAT32);

                    // Resize the input bitmap to match the expected input shape of the model
                    bitmap = Bitmap.createScaledBitmap(bitmap, 416, 416, true);

                    // Create a TensorImage from the resized bitmap
                    TensorImage inputImage = new TensorImage(DataType.FLOAT32);
                    inputImage.load(bitmap);
                    //inputImage = TensorImage.fromBitmap(bitmap);


                    ByteBuffer byteBuffer=inputImage.getBuffer();
                    // Load the pixel values from the TensorImage into the input buffer
                    inputFeature0.loadBuffer(byteBuffer);




                    //bitmap = Bitmap.createScaledBitmap(bitmap, 32 , 32, true);
                    // Creates inputs for reference.
                    //TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
                    // TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 416, 416, 3}, DataType.FLOAT32);

                    //DEBUG
                   // Log.d("shape", byteBuffer.toString());
                   // Log.d("shape", inputFeature0.buffer.toString());

                    //inputFeature0.loadBuffer(TensorImage.fromBitmap(bitmap).getBuffer());

                    // Runs model inference and gets result.
                    Yolov3416Fp16.Outputs outputs = model.process(inputFeature0);

                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    result.setText(getMax(outputFeature0.getFloatArray())+" ");


//                    TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();

                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }


            }
        });

    }
    int getMax(float[] arr){
        int max=0;
        for(int i=0; i<arr.length; i++){
            if(arr[i]> arr[max]) max=i;

        }
        return max;
    }

    void getPermission(){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if(checkSelfPermission(Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED){
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA} , 11);

            }
        }

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode==11){
            if(grantResults.length>0){
                if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
                    this.getPermission();
                }

            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        this.data = data;
        if(requestCode==10){
            Uri uri = data != null ? data.getData() : null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                imageview.setImageBitmap(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        else if(requestCode == 12 ) {
            assert data != null;
            bitmap = (Bitmap) data.getExtras().get("data");
            imageview.setImageBitmap(bitmap);

        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
