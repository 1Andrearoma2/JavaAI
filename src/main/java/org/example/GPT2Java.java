package org.example;

import ai.onnxruntime.*;
import java.util.*;
import static org.example.Methods.*;

public class GPT2Java {
    public static Map<Integer, String> vocab;

    public static void main(String[] args) throws Exception {
        String vocabPath = "C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\vocab.json";
        String modelPath = "C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\onnx\\decoder_model.onnx"; // Qui si puó scegliere il modello
        vocab = loadVocabulary(vocabPath);

        // Qui creiamo la sessione onnx con il modello scelto
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, options);

        // Qui inizializiamo il prompt da dare al modello
        List<Long> inputTokens = new ArrayList<>();
        for (long token : tokenizePrompt()) {
            inputTokens.add(token);
        }

        int maxTokens = 20;  // Questi sono usati per gestire la potenza del modello
        long eosToken = 50256;

        for (int step = 0; step < maxTokens; step++) {
            long[] inputArray = inputTokens.stream().mapToLong(Long::longValue).toArray();

            // Creiamo l'attention_mask
            long[] attentionMaskArray = new long[inputArray.length];
            Arrays.fill(attentionMaskArray, 1);

            // Creiamo i tensori onnx per l'input
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, new long[][]{inputArray});
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, new long[][]{attentionMaskArray});

            // Mappa degli input per onnx
            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("input_ids", inputTensor);
            feeds.put("attention_mask", attentionMaskTensor);

            // Avviamo la sessione
            OrtSession.Result result = session.run(feeds);
            float[][][] outputData = (float[][][]) result.get(0).getValue();

            // Prendiamo il token piú probabile
            float[] logits = outputData[0][outputData[0].length - 1];
            int nextToken = argmax(applySoftmax(logits));

            // Aggiungiamo il token alla sequenza
            inputTokens.add((long) nextToken);
            String tokenToString = decodeToken(nextToken);
            tokenToString = tokenToString.replace("0x0A", " ").replace("Ċ", " ");
            System.out.println("Token generato: " + nextToken + " → " + tokenToString);

            // Interrompiamo se il token é uguale al eosToken
            if (nextToken == eosToken) {
                break;
            }
        }

        // Decodifiamo la sequenza finale
        String generatedText = decodeTokens(inputTokens);
        generatedText = generatedText.replace("<0x0A>", " ").replace("Ċ", " ");
        System.out.println("\nTesto generato:\n" + generatedText);
    }
}
