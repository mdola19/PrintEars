#include <driver/i2s.h>

#define I2S_WS 25
#define I2S_SCK 33
#define I2S_SD 32

#define I2S_SAMPLE_RATE 16000
#define I2S_BUFFER_SIZE 256
#define MAX_CLIP_LENGTH 40000  // 5 seconds at 16 kHz with int16_t

// bool recording = false;
// unsigned long clipStart = 0;

// int16_t *clipData = nullptr;
// int clipIndex = 0;

void setup() {
  Serial.begin(115200);
  delay(500);
  // Serial.println("Ready. Type 'r' to start, 's' to stop and print as CSV column.");

  // Allocate memory dynamically on heap
  // clipData = (int16_t *)malloc(MAX_CLIP_LENGTH * sizeof(int16_t));
  // Serial.print("Free heap: ");
  // Serial.println(ESP.getFreeHeap());
  // if (!clipData) {
  //   Serial.println("Memory allocation failed.");
  //   while (true);
  // }

  // I2S configuration
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = I2S_BUFFER_SIZE,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}


void loop() {
  int16_t audio_samples[I2S_BUFFER_SIZE];
  size_t bytes_read;

  i2s_read(I2S_NUM_0, audio_samples, sizeof(audio_samples), &bytes_read, portMAX_DELAY);

  for (int i = 0; i < bytes_read / sizeof(int16_t); i++) {
    Serial.println(audio_samples[i]);
  }
}


// void loop() {
//   if (Serial.available()) {
//     char input = Serial.read();
//     if (input == 'r') {
//       recording = true;
//       clipStart = millis();
//       clipIndex = 0;
//       Serial.println("Recording...");
//     } else if (input == 's') {
//       recording = false;
//       unsigned long clipDuration = millis() - clipStart;

//       // Start marker to help Python recognize clip start
//       Serial.println("===CLIP_START===");

//       Serial.print("Clip Length (ms): ");
//       Serial.println(clipDuration);
//       Serial.println("Copy this column into your CSV:");

//       for (int i = 0; i < clipIndex; i++) {
//         Serial.println(clipData[i]);
//       }

//       // End marker to help Python know when clip ends
//       Serial.println("===CLIP_END===");
//     }
//   }

//   if (recording && clipIndex < MAX_CLIP_LENGTH) {
//     int16_t audio_samples[I2S_BUFFER_SIZE];
//     size_t bytes_read;

//     i2s_read(I2S_NUM_0, audio_samples, sizeof(audio_samples), &bytes_read, portMAX_DELAY);

//     for (int i = 0; i < bytes_read / sizeof(int16_t); i++) {
//       if (clipIndex < MAX_CLIP_LENGTH) {
//         clipData[clipIndex++] = audio_samples[i];
//       }
//     }
//   }
// }
