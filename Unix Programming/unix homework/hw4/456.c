#include <stdio.h>
#include <string.h>
int main()
{
    //getchar();
    size_t size;
    char *line = NULL;
    char * pch;
    if (getline(&line, &size, stdin) == -1) {
        printf("No line\n");
    } else {
        printf("%s\n", line);
        
        char * pch;
        pch = strtok (line," ,.-");
        while (pch != NULL)
        {
            printf ("%s\n",pch);
            pch = strtok (NULL, " ,.-");
        }
        
        
    }
    return 0;
}