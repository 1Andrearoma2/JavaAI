package org.example;

import java.util.*;
import static org.example.GPT2Java.vocab;

public class Tokenizer {
    /**
     * Metodo usato per trasformare il prompt in comprensibile per il modello traducendolo in tokens
     * @param prompt String da trasformare in tokens
     * @return Il prompt tradotto in tokens sotto forma di array di long
     */
    public static long[] tokenize(String prompt) {
        List<Integer> tokenList = new ArrayList<>();
        String[] words = prompt.split(" ");

        for (String word : words) {
            processWord(word, tokenList);
            tokenList.add(13); // Aggiunge il separatore tra parole
        }

        // Rimuove l'ultimo separatore 13 se presente
        if (!tokenList.isEmpty() && tokenList.get(tokenList.size() - 1) == 13L) {
            tokenList.remove(tokenList.size() - 1);
        }

        return tokenList.stream().mapToLong(Integer::longValue).toArray();
    }

    /**
     * Metodo che traduce parola per parola. Se essa non viene trovata viene divisa a meta'
     * o al massimo carattere per carattere
     * @param word
     * @param tokenList
     */
    private static void processWord(String word, List<Integer> tokenList) {
        Integer token = findInVocab(word);
        if (token != null) {
            tokenList.add(token);
            return;
        }

        // Divisione a metà
        int mid = word.length() / 2;
        String part1 = word.substring(0, mid);
        String part2 = word.substring(mid);

        Integer token1 = findInVocab(part1);
        Integer token2 = findInVocab(part2);

        if (token1 != null && token2 != null) {
            tokenList.add(token1);
            tokenList.add(token2);
            return;
        }

        // Divisione carattere per carattere
        for (char c : word.toCharArray()) {
            Integer charToken = findInVocab(String.valueOf(c));
            if (charToken != null) {
                tokenList.add(charToken);
            }
        }
    }

    /**
     * Metodo usato per cercare la parola all'interno della mappa
     * @param word da ricercare nella mappa
     * @return Il token dell'input se viene trovato nella mappa, altrimenti null
     */
    private static Integer findInVocab(String word) {
        for (Map.Entry<Integer, String> entry : vocab.entrySet()) {
            if (entry.getValue().equals(word)) {
                return entry.getKey();
            }
        }
        return null;
    }
}
