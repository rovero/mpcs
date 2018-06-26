#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>
#include <netdb.h>
#include <string.h>
#include <stdlib.h>
#define MAXSIZE 50
int main( int argc, char * argv[] )
{
	int sock, conn;
	struct sockaddr_in address;
	int n, len;
	struct hostent * host;
    char *buf;
	char hostname[64];
	int x = 0x12345678;
	struct in_addr inaddr;

	if( argc != 3 )
	{
		printf("USAGE: %s hostname\n",argv[0]);
		exit(-1);
	}

	if( inet_aton(argv[1], &inaddr))
		host = gethostbyaddr((char *) &inaddr, sizeof(inaddr), AF_INET);
	else
		host = gethostbyname(argv[1]);

	if ( !host )
	{
		printf("no host\n");
		exit(1);
	}

	if( (sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		perror("socket:");
		exit(-1);
	}

/*	memset(&address, 0, sizeof(struct sockaddr_in)); */
	address.sin_family = AF_INET;
	address.sin_port = htons(atoi(argv[2]));
	memcpy(&address.sin_addr, host->h_addr_list[0], sizeof(address.sin_addr));
/*	len = sizeof(struct sockaddr_in); */

	if (connect(sock, (struct sockaddr *) &address, sizeof(address)) < 0 )
	{
		perror("connect:");
		exit(-1);
	}

    char str1[50],str2[200];
    
    printf("Enter word: ");
    scanf("%s", str1);
    
    
    //printf("the sizeo of the word is %d\n",strlen(str1));
	if( write(sock, str1, sizeof(str1)) < 0 )
	{
		perror("write:");
		exit(-1);
	}
	//write(1,&x,sizeof(int));
    memset(str2,0,strlen(str2));
    if( read(sock, str2, sizeof(str2)) < 0 )
    {
        perror("read:");
        exit(-1);
    }
    char haha[] = "yes";
    if(strcmp(str2,haha)==0){
        printf("The word %s is spelled correctly\n",str1);
    }
    else{
        printf("The word is spelled uncorrectly\n");
        //read(sock, str2, sizeof(str2));
        
        if( read(sock, str2, sizeof(str2)) < 0 )
        {
            perror("read:");
            exit(-1);
        }
        printf("%s",str2);
    }
    //printf("the receive buffer is %s\n",str1);
    
	close(sock);
	exit(0);
}





