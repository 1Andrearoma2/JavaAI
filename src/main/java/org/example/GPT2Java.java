package org.example;

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class GPT2Java {

    private static Map<Integer, String> vocab;

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

        int maxTokens = 50;  // Questi sono usati per gestire la potenza del modello
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

    // Metodo per gestire il prompt
    private static long[] tokenizePrompt() {
        return new long[]{22110, 13, 275, 13, 20841, 2049, 13, 14958, 2929, 2172, 29973};  // 29973 = ?
    }

    // Funzione per caricare il vocabolario dal vocab.json
    private static Map<Integer, String> loadVocabulary(String filePath) {
        Map<Integer, String> vocabMap = new HashMap<>();
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            Map<String, Integer> rawVocab = objectMapper.readValue(new File(filePath), Map.class);

            // Visto che il file .json é String-Int e a noi serve Int-String, invertiamo la mappa
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
    // Il softmax serve per convertire i logits in probabilita, cioe trasforma
    // numeri grezzi in numeri piu facili da usare
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

    // Funzione per trovare l'indice massimo di un array
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
