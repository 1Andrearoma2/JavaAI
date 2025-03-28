package org.example;

import java.util.*;

public class Tokenizer {
    public static long[] tokenize(String prompt, Map<Integer, String> vocab) {
        List<Integer> tokenList = new ArrayList<>();
        String[] words = prompt.split(" ");

        for (String word : words) {
            processWord(word, vocab, tokenList);
            tokenList.add(13); // Aggiunge il separatore tra parole
        }

        // Rimuove l'ultimo separatore 13 se presente
        if (!tokenList.isEmpty() && tokenList.get(tokenList.size() - 1) == 13L) {
            tokenList.remove(tokenList.size() - 1);
        }

        return tokenList.stream().mapToLong(Integer::longValue).toArray();
    }

    private static void processWord(String word, Map<Integer, String> vocab, List<Integer> tokenList) {
        Integer token = findInVocab(word, vocab);
        if (token != null) {
            tokenList.add(token);
            return;
        }

        // Divisione a metà
        int mid = word.length() / 2;
        String part1 = word.substring(0, mid);
        String part2 = word.substring(mid);

        Integer token1 = findInVocab(part1, vocab);
        Integer token2 = findInVocab(part2, vocab);

        if (token1 != null && token2 != null) {
            tokenList.add(token1);
            tokenList.add(token2);
            return;
        }

        // Divisione in lettere
        for (char c : word.toCharArray()) {
            Integer charToken = findInVocab(String.valueOf(c), vocab);
            if (charToken != null) {
                tokenList.add(charToken);
            }
        }
    }

    private static Integer findInVocab(String word, Map<Integer, String> vocab) {
        for (Map.Entry<Integer, String> entry : vocab.entrySet()) {
            if (entry.getValue().equals(word)) {
                return entry.getKey();
            }
        }
        return null;
    }
}
