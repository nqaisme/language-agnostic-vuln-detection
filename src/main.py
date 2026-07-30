from model import extractor

c_snippets = [
    """
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
""",

    """
#include <stdio.h>

int main() {
    int age = 20;
    float height = 1.75;
    char grade = 'A';

    printf("Age: %d\\n", age);
    printf("Height: %.2f\\n", height);
    printf("Grade: %c\\n", grade);

    return 0;
}
""",

    """
#include <stdio.h>

int main() {
    int number = 10;

    if (number > 0) {
        printf("Positive\\n");
    } 
    else if (number < 0) {
        printf("Negative\\n");
    } 
    else {
        printf("Zero\\n");
    }

    return 0;
}
""",

    """
#include <stdio.h>

int main() {

    for (int i = 0; i < 10; i++) {
        printf("%d\\n", i);
    }

    return 0;
}
""",

    """
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {

    int result = add(5, 3);

    printf("%d\\n", result);

    return 0;
}
""",

    """
#include <stdio.h>

int main() {

    int value = 100;

    int *ptr = &value;

    printf("Value: %d\\n", *ptr);

    return 0;
}
""",

    """
#include <stdio.h>

int main() {

    int numbers[5] = {1,2,3,4,5};

    for(int i = 0; i < 5; i++) {
        printf("%d\\n", numbers[i]);
    }

    return 0;
}
""",

    """
#include <stdio.h>
#include <stdlib.h>

int main() {

    int *ptr;

    ptr = malloc(sizeof(int));

    *ptr = 50;

    printf("%d\\n", *ptr);

    free(ptr);

    return 0;
}
""",

    """
#include <stdio.h>

int main() {

    char buffer[10];

    gets(buffer);

    printf("%s\\n", buffer);

    return 0;
}
""",

    """
#include <stdio.h>

int main() {

    char buffer[20];

    fgets(buffer, sizeof(buffer), stdin);

    printf("%s", buffer);

    return 0;
}
""",

    """
#include <stdio.h>

int main()
{
    FILE *file;

    char text[100];

    file = fopen("data.txt", "r");

    if(file != NULL) {
        fgets(text, sizeof(text), file);
        printf("%s", text);
        fclose(file);
    }

    return 0;
}
""",

    """
#include <stdlib.h>

int main()
{
    char *buffer = malloc(100);

    free(buffer);

    free(buffer);

    return 0;
}
""",

    """
#include <stdio.h>
#include <stdlib.h>

int main()
{
    int *ptr = malloc(sizeof(int));

    *ptr = 10;

    free(ptr);

    printf("%d", *ptr);

    return 0;
}
""",

    """
#include <stdio.h>

int main()
{
    int *ptr = NULL;

    printf("%d", *ptr);

    return 0;
}
""",

    """
#include <stdio.h>

int factorial(int n)
{
    if(n <= 1)
        return 1;

    return n * factorial(n-1);
}

int main()
{
    printf("%d", factorial(5));

    return 0;
}
"""
]
hh = extractor(model_name='microsoft/graphcodebert-base')
zh = hh(c_snippets)

print(zh.shape)
print(len(c_snippets))