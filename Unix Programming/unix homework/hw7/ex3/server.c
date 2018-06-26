#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sem.h>
#include <sys/types.h>
#include <unistd.h>
 union semun {
     int val;
     struct semid_ds *buf;
     unsigned short int *array;
    };
//#include <sys/sem.h>

/*
* Shared memory server.
*/
#define LEN 10
union semun sem_union;
static int sem_id;

static int semaphore_p(void) {
    struct sembuf sem_b;
    sem_b.sem_num = 0;
    sem_b.sem_op = -1; /* P() */
    sem_b.sem_flg = SEM_UNDO;
    if (semop(sem_id, &sem_b, 1) == -1) {
        fprintf(stderr,"semaphore_p failed\n");
        return (0);
    }
    return 1;
}


static int semaphore_v(void) {
    struct sembuf sem_b;
    sem_b.sem_num = 0;
    sem_b.sem_op = 1; /* V() */ sem_b.sem_flg = SEM_UNDO;
    if (semop(sem_id, &sem_b, 1) == -1) {
        fprintf(stderr, "semaphore_v failed\n");
        return(0);
    }
    return(1);
}

static void del_semvalue(void) {
    union semun sem_union;
    if (semctl(sem_id, 0, IPC_RMID, sem_union) == -1) fprintf(stderr, "Failed to delete semaphore\n");
}



static int set_semvalue(void) {
    union semun sem_union;
    sem_union.val = 1;
    if (semctl(sem_id, 0, SETVAL, sem_union) == -1) return(0); return(1);
}
    

int main(int argc, char *argv[])
{
  int shmid;
  char *shm;
  key_t key = IPC_PRIVATE;
  int flag = SHM_R | SHM_W|IPC_CREAT;

  if (argc >= 2) {
    key = atoi(argv[1]);
  }

  shmid = shmget(key, LEN, flag);
  printf("shared memory for key %d: %d\n", key, shmid);
  if (shmid < 0) {
    perror("shmget");
    perror("Try to create this segment\n");
    shmid = shmget(key, LEN, flag | IPC_CREAT);
    if (shmid < 0) {
      perror("shmget | IPC_CREAT");
    }
  }
    
  sem_id = semget((key_t)1236, 1, 0666 | IPC_CREAT);
    set_semvalue();
    if(sem_id==-1){
        perror("fail to create semget\n");
        exit(EXIT_FAILURE);
    }
        
    
    //if (!semaphore_p()) exit(EXIT_FAILURE);

  shm = (char *)shmat(shmid, /*addr*/0, /*flag*/0); 
  if (shm == (char *)-1) {
    perror("shmat");
    exit(1);
  }
    
    struct shmid_ds *buf = malloc(sizeof(struct shmid_ds));
    
    shmctl(shmid,IPC_STAT,buf);
    if(getpid() == buf->shm_cpid){
        sprintf(shm, "1");
    }
  printf("shared memory content: %s\n", shm);
  //char *shm;
    shm = (char *)shmat(shmid, /*addr*/0, /*flag*/0);
    while(1){
        
        /* START of CRITICAL SECTION */
        if (!semaphore_p()) exit(EXIT_FAILURE);
        int num = atoi(shm);
        printf("left count  = %d\n",num);
        if(num>=5){
            
            /* END of CRITICAL SECTION */
            if (!semaphore_v()) exit(EXIT_FAILURE);
            sleep(2);
            continue;
        }
        num++;
        char c[10];
        sprintf(c,"%d",num);
        printf("now count: %s\n",c);
        sprintf(shm,c);
        
        /* END of CRITICAL SECTION */
        if (!semaphore_v()) exit(EXIT_FAILURE);
        sleep(2);
    }
    

    
    
  return 0;
} /* main */
