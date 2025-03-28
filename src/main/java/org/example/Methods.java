package org.example;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.example.GPT2Java.vocab;

public class Methods {

    // Funzione per caricare il vocabolario dal vocab.json
    public static Map<Integer, String> loadVocabulary(String filePath) {
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
    public static String decodeToken(int tokenId) {
        return vocab.getOrDefault(tokenId, "<UNK>");
    }

    // Funzione per decodificare tutta la sequenza
    public static String decodeTokens(List<Long> tokens) {
        StringBuilder sb = new StringBuilder();
        for (long token : tokens) {
            sb.append(decodeToken((int) token));
        }
        return sb.toString();
    }

    // Funzione per applicare softmax
    // Il softmax serve per convertire i logits in probabilita, cioe trasforma
    // numeri grezzi in numeri piu facili da usare
    public static float[] applySoftmax(float[] logits) {
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
    public static int argmax(float[] array) {
        int index = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[index]) {
                index = i;
            }
        }
        return index;
    }

}
