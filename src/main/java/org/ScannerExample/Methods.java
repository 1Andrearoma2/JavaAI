package org.ScannerExample;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.example.GPT2Java.vocab;

public class Methods {

    /**
     * Metodo usato per caricare il vocabolario all'inizio del programma
     * @param filePath file vocab.json in cui sono scritte tutte le corrispondenze
     * @return una mappa che contiene tutte le corrispondenze presenti nel file vocab.json
     */
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

    /**
     * Decodifica il token in stringa cercandolo nella mappa, se non viene trovato
     * ritorna un valore sconosciuto
     * @param tokenId token da decodificare
     * @return la stringa che corrisponde al token
     */
    public static String decodeToken(int tokenId) {
        return vocab.getOrDefault(tokenId, "<UNK>");
    }

    /**
     * Decodifica una lista di token cercandoli nella mappa attraverso il metodo decodeToken
     * @param tokens lista contenente i token da decodificare
     * @return La stringa intera
     */
    public static String decodeTokens(List<Long> tokens) {
        StringBuilder sb = new StringBuilder();
        for (long token : tokens) {
            sb.append(decodeToken((int) token));
        }
        return sb.toString();
    }

    /**
     * Funzione per applicare il softmax ad un array di logits
     * e farli diventare delle probabilita'
     * @param logits array in cui applicare il softmax
     * @return L'array convertito in probabilita'
     */
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

    /**
     * Metodo per trovare l'indice massimo in un array
     * @param array da elaborare
     * @return L'indice massimo
     */
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
