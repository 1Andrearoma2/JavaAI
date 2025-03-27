package org.example;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

import java.io.File;
import java.io.IOException;
import java.util.Map;

public class Tokenizer {
    private static Map<String, Long> vocab;

    public static void main(String[] args) throws IOException {
        // Carica il vocabolario dal file JSON
        loadVocabulary("C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\vocab.json");

        // Definisci il prompt
        String prompt = "I hate niggers";

        // Tokenizza il prompt
        long[] tokenizedPrompt = tokenize(prompt);

        // Stampa i token
        System.out.println("Tokenized Prompt: ");
        for (long token : tokenizedPrompt) {
            System.out.print(token + " ");
        }
    }

    // Funzione per caricare il vocabolario da vocab.json
    private static void loadVocabulary(String vocabFilePath) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        // Usa TypeReference per indicare che il vocabolario è una mappa di stringa -> long
        vocab = objectMapper.readValue(new File(vocabFilePath), new TypeReference<Map<String, Long>>() {});
    }

    // Funzione per tokenizzare il prompt (tradurre ogni parola nel corrispondente ID del vocabolario)
    private static long[] tokenize(String prompt) {
        String[] words = prompt.split(" ");
        long[] tokenizedPrompt = new long[words.length];

        for (int i = 0; i < words.length; i++) {
            String word = words[i].replaceAll("[^\\w\\s]", "");  // Rimuove eventuali punteggiature
            if (vocab.containsKey(word)) {
                tokenizedPrompt[i] = vocab.get(word);
            } else {
                // Se non esiste, puoi decidere come trattare la parola sconosciuta
                // E.g., aggiungere un ID per "unknown" o saltarla
                tokenizedPrompt[i] = vocab.get("<unk>");  // Sostituire con un ID per parole sconosciute, se presente
            }
        }

        return tokenizedPrompt;
    }
}
