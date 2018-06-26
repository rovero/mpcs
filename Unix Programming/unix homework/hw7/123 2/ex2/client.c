#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
* Shared memory client.
*/

int main(int argc, char *argv[])
{
  int shmid;
  char *shm;
  key_t key;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s key\n", argv[0]);
    exit(2);
  }
  key = atoi(argv[1]);

  printf("Trying shared memory %d\n", key);
  shmid = shmget(key, 1000, SHM_R | SHM_W);
  if (shmid < 0) {
    perror("shmget");
    shmid = key;
  }
  else {
    printf("shared memory: %d\n", shmid);
  } 

  shm = (char *)shmat(shmid, /*addr*/0, /*flag*/0); 
  if (shm == (char *)-1) {
    perror("shmat");
    exit(1);
  }
  printf("shared memory: %p\n", shm);
    printf("convert char* to int %d\n",atoi(shm));
  if (shm != 0) {
    printf("shared memory content: %s\n", shm);
  }
  
    
    while(1){
        //shm = (char *)shmat(shmid, /*addr*/0, /*flag*/0);
        int num = atoi(shm);
        if(num<=0){
            
            fprintf(stdout,"Blasted!! I knew I should have gone to T---t.\n");
            sleep(3);
            continue;
        }
        num--;
        printf("num = %d\n",num);
        char c[10]={0};
        
        sprintf(c,"%d",num);
        sprintf(shm,c);
        sleep(1);
    }
    
  return 0;
} /* main */
