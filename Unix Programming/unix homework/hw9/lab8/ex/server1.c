#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>
#define MAXSIZE 256

struct spellings{
    int similarity;
    char word[MAXSIZE];
};
typedef struct spellings spellings;

void dictionary(const char* str, int *result, spellings *vec[]);


int main( int argc, char * argv[] )
{
    int sock, conn;
    struct sockaddr_in serveradd;
    struct sockaddr_in clientadd;
    int n, len;
    char buf[BUFSIZ];
    
    sock = socket(AF_INET, SOCK_STREAM, 0);
    
    serveradd.sin_family = AF_INET;
    //inet_aton(argv[1], serveradd.sin_addr.s_addr);
    serveradd.sin_addr.s_addr = inet_addr("127.0.0.1");//argv[1]
    //serveradd.sin_addr.s_addr = htonl(INADDR_ANY);
    
    serveradd.sin_port = htons(atoi(argv[1]));
    len = sizeof(struct sockaddr_in);
    
    if( bind(sock, (struct sockaddr *) & serveradd, sizeof(serveradd)))
    {
        perror("bind:");
        exit(-1);
    }
    
    if( listen(sock,5) )
    {
        perror("bind:");
        exit(-1);
    }
    
    if ((conn = accept(sock, (struct sockaddr *) &clientadd, &len)) < 0 )
    {
        perror("bind:");
        exit(-1);
    }
    spellings *vec[10];
    //vec = (spellings**) malloc(10*sizeof(spellings*));
    //printf("the length of vec is %d\n",strlen(vec));
    int i=0;
    for(i=0;i<10;i++){
        vec[i] = (spellings *)malloc(sizeof(spellings)); // malloc(sizeof(struct spellings));
        vec[i]->similarity = -1;
    }
    while( (n = recv(conn, buf, sizeof(buf), 0)) > 0){
        printf("the word server receives is %s\n",buf);
        int res=0;
        dictionary(buf,&res,vec);
        if(res){
            //int send(int socket, char *buf, int len, int flags);
            char haha[]="yes";
            send(conn,haha,strlen(haha),0);
            //send(conn,haha,strlen(haha),0);
        }
        else{
            char haha[]="no";
            
            send(conn,haha,strlen(haha),0);
            char hehe[200];
            char empty[] = "  ";
            int i=0, len = 0;
            for(i=0;i<10;i++){
                if(vec[i]->similarity==-1)
                    continue;
                
                printf("server side: %s",vec[i]->word);
                memcpy(hehe+len, vec[i]->word, strlen(vec[i]->word));
                len+=strlen(vec[i]->word);
                
                //memcpy(hehe+len, empty, strlen(empty));
            }
            if(send(conn,hehe,200,0)!=200)
		printf("wrong in sending\n");
        }
    }
    
    close(conn);
    close(sock);
    exit(0);
}


int compute_similarity(const char *str1, const char *str2){
    int res = 0;
    unsigned int i=0;
    int len = strlen(str1)<strlen(str2)? strlen(str1):strlen(str2);
    while(i<len){
        res+=abs(*(str1+i)-*(str2+i));
        i++;
    }
    return res;
}

void dictionary(const char* str, int *result, spellings *vec[])
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    
    
    //struct spellings vec[10] = {0};
    fp = fopen("/usr/share/dict/words", "r"); // /Users/Gaopeng/Desktop/unix_hw/word // /usr/share/dict/words
    if (fp == NULL)
        exit(EXIT_FAILURE);
    
    while ((read = getline(&line, &len, fp)) != -1) {
        //printf("Retrieved line of length %zu :\n", read);
        char *temp = malloc(20*sizeof(char));
        memcpy(temp, line, strlen(line)-1);
        if(strcmp(str,temp)==0){
            *result = 1;
            return;
        }
        else{
            int i=0, res = compute_similarity(str,line);
            if(abs(strlen(str)-strlen(line))>2)
                continue;
            int big_sim_index;
            unsigned int sim_val = 0;
            int jump = 0;
            if(res>=27)
                continue;
            for(i=0;i<10;i++){
                if(vec[i]->similarity==-1){
                    vec[i]=malloc(sizeof(struct spellings));
                    memcpy(vec[i]->word,line,strlen(line)*sizeof(char));
                    vec[i]->similarity = res;
                    jump = 1;
                    break;
                }
                if(sim_val<vec[i]->similarity){
                    sim_val = vec[i]->similarity;
                    big_sim_index = i;
                }
            }
            
            if(jump==1)
                continue;
            vec[big_sim_index]->similarity = res;
            memset(vec[big_sim_index]->word,0,strlen(vec[big_sim_index]->word));
            memcpy(vec[big_sim_index]->word,line,strlen(line)*sizeof(char));
        }
        //printf("%s", line);
        free(temp);
    }// end of while
    
    fclose(fp);
    if (line)
        free(line);
    return;
}






