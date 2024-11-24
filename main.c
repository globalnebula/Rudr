#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

// Constants
#define MAX_WORDS 1000
#define MAX_TEXT_LEN 1024
#define MAX_VOCAB_SIZE 500
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define REGULARIZATION 0.001
#define EPOCHS 1000

// Function Prototypes
void preprocess_text(char* text);
char** tokenize(char* text, int* word_count);
char** build_vocabulary(char* file_path, int* vocab_size);
int* create_bow_vector(char* text, char** vocabulary, int vocab_size);
double sigmoid(double x);
double sigmoid_derivative(double x);
void train_model(double* weights, double* bias, char** vocabulary, int vocab_size, char* train_file);
double test_model(double* weights, double bias, char** vocabulary, int vocab_size, char* test_file);
void save_model(double* weights, double bias, int vocab_size, char* model_file);
void load_model(double* weights, double* bias, int vocab_size, char* model_file);
void predict(double* weights, double bias, char** vocabulary, int vocab_size);

// Preprocess text (to lowercase, remove punctuation)
void preprocess_text(char* text) {
    for (int i = 0; text[i]; i++) {
        if (ispunct(text[i])) text[i] = ' ';
        text[i] = tolower(text[i]);
    }
}

// Tokenize text
char** tokenize(char* text, int* word_count) {
    char** tokens = malloc(MAX_WORDS * sizeof(char*));
    char* token = strtok(text, " ");
    int count = 0;

    while (token != NULL && count < MAX_WORDS) {
        tokens[count++] = strdup(token);
        token = strtok(NULL, " ");
    }
    *word_count = count;
    return tokens;
}

// Build vocabulary from training file
char** build_vocabulary(char* file_path, int* vocab_size) {
    FILE* file = fopen(file_path, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char** vocabulary = malloc(MAX_VOCAB_SIZE * sizeof(char*));
    int vocab_count = 0;
    char line[MAX_TEXT_LEN];

    while (fgets(line, MAX_TEXT_LEN, file)) {
        char* text = strtok(line, ",");
        preprocess_text(text);

        int word_count;
        char** tokens = tokenize(text, &word_count);

        for (int i = 0; i < word_count; i++) {
            int found = 0;
            for (int j = 0; j < vocab_count; j++) {
                if (strcmp(tokens[i], vocabulary[j]) == 0) {
                    found = 1;
                    break;
                }
            }
            if (!found && vocab_count < MAX_VOCAB_SIZE) {
                vocabulary[vocab_count++] = strdup(tokens[i]);
            }
        }

        for (int i = 0; i < word_count; i++) free(tokens[i]);
        free(tokens);
    }

    fclose(file);
    *vocab_size = vocab_count;
    return vocabulary;
}

// Create Bag of Words vector
int* create_bow_vector(char* text, char** vocabulary, int vocab_size) {
    int* bow_vector = calloc(vocab_size, sizeof(int));
    preprocess_text(text);

    int word_count;
    char** tokens = tokenize(text, &word_count);

    for (int i = 0; i < word_count; i++) {
        for (int j = 0; j < vocab_size; j++) {
            if (strcmp(tokens[i], vocabulary[j]) == 0) {
                bow_vector[j]++;
                break;
            }
        }
    }
    for (int i = 0; i < word_count; i++) free(tokens[i]);
    free(tokens);
    return bow_vector;
}

// Sigmoid Activation
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Sigmoid Derivative
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Train Model
void train_model(double* weights, double* bias, char** vocabulary, int vocab_size, char* train_file) {
    FILE* file = fopen(train_file, "r");
    if (!file) {
        perror("Error opening training file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_TEXT_LEN];
    double* velocity = calloc(vocab_size, sizeof(double));
    if (!velocity) {
        perror("Memory allocation failed for velocity");
        exit(EXIT_FAILURE);
    }


    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        rewind(file);
        while (fgets(line, MAX_TEXT_LEN, file)) {
            char* text = strtok(line, ",");
            int label = atoi(strtok(NULL, ","));

            int* bow_vector = create_bow_vector(text, vocabulary, vocab_size);

            double z = 0;
            for (int i = 0; i < vocab_size; i++) {
                z += bow_vector[i] * weights[i];
            }
            z += *bias;
            double prediction = sigmoid(z);

            double error = label - prediction;
            double gradient = error * sigmoid_derivative(prediction);

            for (int i = 0; i < vocab_size; i++) {
                velocity[i] = MOMENTUM * velocity[i] + LEARNING_RATE * (gradient * bow_vector[i] - REGULARIZATION * weights[i]);
                weights[i] += velocity[i];
            }
            *bias += LEARNING_RATE * gradient;

            free(bow_vector);
        }
        printf("Epoch %d completed\n", epoch + 1);
    }

    fclose(file);
}

// Test Model
double test_model(double* weights, double bias, char** vocabulary, int vocab_size, char* test_file) {
    FILE* file = fopen(test_file, "r");
    if (!file) {
        perror("Error opening test file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_TEXT_LEN];
    int correct = 0, total = 0;

    while (fgets(line, MAX_TEXT_LEN, file)) {
        char* text = strtok(line, ",");
        int label = atoi(strtok(NULL, ","));

        int* bow_vector = create_bow_vector(text, vocabulary, vocab_size);

        double z = 0;
        for (int i = 0; i < vocab_size; i++) {
            z += bow_vector[i] * weights[i];
        }
        z += bias;
        double prediction = sigmoid(z) > 0.5 ? 1 : 0;

        if (prediction == label) correct++;
        total++;

        free(bow_vector);
    }

    fclose(file);
    return (double)correct / total;
}

// Save Model
void save_model(double* weights, double bias, int vocab_size, char* model_file) {
    FILE* file = fopen(model_file, "w");
    if (!file) {
        perror("Error saving model");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < vocab_size; i++) {
        fprintf(file, "%lf\n", weights[i]);
    }
    fprintf(file, "%lf\n", bias);
    fclose(file);
}

// Load Model
void load_model(double* weights, double* bias, int vocab_size, char* model_file) {
    FILE* file = fopen(model_file, "r");
    if (!file) {
        perror("Error loading model");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < vocab_size; i++) {
        fscanf(file, "%lf", &weights[i]);
    }
    fscanf(file, "%lf", bias);
    fclose(file);
}

// Predict new text
void predict(double* weights, double bias, char** vocabulary, int vocab_size) {
    char input[MAX_TEXT_LEN];
    printf("Enter text to classify: ");
    fgets(input, MAX_TEXT_LEN, stdin);

    int* bow_vector = create_bow_vector(input, vocabulary, vocab_size);

    double z = 0;
    for (int i = 0; i < vocab_size; i++) {
        z += bow_vector[i] * weights[i];
    }
    z += bias;

    double prediction = sigmoid(z) > 0.5 ? 1 : 0;
    printf("Prediction: %d\n", (int)prediction);

    free(bow_vector);
}

// Main Function
int main() {
    char train_file[] = "train.csv";
    char test_file[] = "test.csv";
    char model_file[] = "model.dat";

    int vocab_size;
    char** vocabulary = build_vocabulary(train_file, &vocab_size);

    double* weights = calloc(vocab_size, sizeof(double));
    double bias = 0;

    train_model(weights, &bias, vocabulary, vocab_size, train_file);
    double accuracy = test_model(weights, bias, vocabulary, vocab_size, test_file);
    printf("Test Accuracy: %.2f%%\n", accuracy * 100);

    save_model(weights, bias, vocab_size, model_file);
    load_model(weights, &bias, vocab_size, model_file);

    predict(weights, bias, vocabulary, vocab_size);

    for (int i = 0; i < vocab_size; i++) free(vocabulary[i]);
    free(vocabulary);
    free(weights);

    return 0;
}
