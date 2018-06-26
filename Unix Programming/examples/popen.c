#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main()
{
        FILE * fp;
        char buffer[BUFSIZ+1];
        sprintf(buffer, "zucchini\nbanana\napple\n");
        fp = popen("sort", "w");
        if( fp != NULL )
        {
                fwrite(buffer, sizeof(char), strlen(buffer), fp);
                pclose(fp);
                exit(EXIT_SUCCESS);
        }
        exit(EXIT_FAILURE);
}

