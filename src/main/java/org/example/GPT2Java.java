package org.example;

import ai.onnxruntime.*;
import org.apache.commons.lang3.ArrayUtils;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class GPT2Java {
    private static Map<Integer, String> vocab;  // Dizionario per la decodifica

    public static void main(String[] args) throws Exception {
        // Carica il vocabolario da vocab.json
        vocab = loadVocabulary("C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\vocab.json");

        // Carica il modello ONNX
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession("C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\onnx\\decoder_model.onnx", options);

        // Definisci il prompt iniziale
        String prompt = "What is machine learning?";
        List<Long> inputTokens = new ArrayList<>();
        for (long token : tokenizePrompt(prompt)) {
            inputTokens.add(token);
        }

        int maxTokens = 20;  // Numero massimo di token da generare
        long eosToken = 50256;  // Token di fine sequenza

        for (int step = 0; step < maxTokens; step++) {
            long[] inputArray = inputTokens.stream().mapToLong(Long::longValue).toArray();

            // Crea la attention_mask
            long[] attentionMaskArray = new long[inputArray.length];
            Arrays.fill(attentionMaskArray, 1);

            // Crea i tensori ONNX per l'input
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, new long[][]{inputArray});
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, new long[][]{attentionMaskArray});

            // Mappa di input per ONNX
            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("input_ids", inputTensor);
            feeds.put("attention_mask", attentionMaskTensor);

            // Esegui l'inferenza
            OrtSession.Result result = session.run(feeds);
            float[][][] outputData = (float[][][]) result.get(0).getValue();

            // Ottieni il token più probabile
            float[] logits = outputData[0][outputData[0].length - 1];
            int nextToken = argmax(applySoftmax(logits));

            // Aggiungi il token alla sequenza
            inputTokens.add((long) nextToken);
            String tokenToString = decodeToken(nextToken);
            tokenToString = tokenToString.replace("Ġ", " ").replace("Ċ", " ");
            System.out.println("Token generato: " + nextToken + " → " + tokenToString);

            // Interrompi se viene generato il token di fine sequenza
            if (nextToken == eosToken) {
                break;
            }
        }

        // Decodifica la sequenza finale
        String generatedText = decodeTokens(inputTokens);
        generatedText = generatedText.replace("Ġ", " ").replace("Ċ", " ");
        System.out.println("\nTesto generato:\n" + generatedText);
    }

    // Funzione per tokenizzare il prompt (da sostituire con una vera tokenizzazione)
    private static long[] tokenizePrompt(String prompt) {
        return new long[]{28851, 8988, 521, 685, 8155, 17607};  // "What is machine learning?"
    }

    // Funzione per caricare il vocabolario da vocab.json
    private static Map<Integer, String> loadVocabulary(String filePath) {
        Map<Integer, String> vocabMap = new HashMap<>();
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            Map<String, Integer> rawVocab = objectMapper.readValue(new File(filePath), Map.class);

            // Convertire la mappa da <String, Integer> a <Integer, String>
            for (Map.Entry<String, Integer> entry : rawVocab.entrySet()) {
                vocabMap.put(entry.getValue(), entry.getKey());
            }

            System.out.println("Vocabolario caricato con " + vocabMap.size() + " parole.");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return vocabMap;
    }

    // Funzione per decodificare un token
    private static String decodeToken(int tokenId) {
        return vocab.getOrDefault(tokenId, "<UNK>");
    }

    // Funzione per decodificare tutta la sequenza
    private static String decodeTokens(List<Long> tokens) {
        StringBuilder sb = new StringBuilder();
        for (long token : tokens) {
            sb.append(decodeToken((int) token));
        }
        return sb.toString();
    }

    // Funzione per applicare softmax
    private static float[] applySoftmax(float[] logits) {
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            maxLogit = Math.max(maxLogit, logit);
        }

        float sumExp = 0;
        for (int i = 0; i < logits.length; i++) {
            logits[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExp += logits[i];
        }

        for (int i = 0; i < logits.length; i++) {
            logits[i] /= sumExp;
        }
        return logits;
    }

    // Funzione per trovare l'indice del massimo
    private static int argmax(float[] array) {
        int index = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[index]) {
                index = i;
            }
        }
        return index;
    }
}
