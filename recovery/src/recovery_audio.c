#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_bit_txt> <output_bin> <pool(1-6)>\n", argv[0]);
        return 2;
    }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    int pool = atoi(argv[3]);

    size_t audio_len = 0;

    switch (pool) {
        case 1: audio_len = 1320907; break;
        case 2: audio_len = 1320646; break;
        case 3: audio_len = 1320986; break;
        case 4: audio_len = 1320237; break;
        case 5: audio_len = 1321010; break;
        case 6: audio_len = 1320580; break;
        default:
            fprintf(stderr, "Error: pool must be 1..6\n");
            return 2;
    }

    size_t num_chars = audio_len * 8;

    char *dec_results = (char *)malloc(num_chars);
    if (dec_results == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    unsigned char *audio = (unsigned char *)calloc(audio_len, 1);
    if (audio == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(dec_results);
        return 1;
    }

    FILE *file = fopen(in_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening input file\n");
        free(dec_results);
        free(audio);
        return 1;
    }

    size_t result = fread(dec_results, 1, num_chars, file);
    if (result != num_chars) {
        fprintf(stderr, "Error reading file or file is too short\n");
        free(dec_results);
        free(audio);
        fclose(file);
        return 1;
    }
    fclose(file);

    for (size_t i = 0; i < audio_len; i++) {
        for (size_t j = 8 * i; j < 8 * (i + 1); j++) {
            audio[i] += (unsigned char)((dec_results[j] - '0') * (1 << (7 - (j % 8))));
        }
    }

    FILE *fpout = fopen(out_path, "wb");
    if (fpout == NULL) {
        fprintf(stderr, "Error opening output file\n");
        free(dec_results);
        free(audio);
        return 1;
    }

    fwrite(audio, 1, audio_len, fpout);
    fclose(fpout);

    free(dec_results);
    free(audio);

    return 0;
}
