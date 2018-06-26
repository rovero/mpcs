       #include <sys/types.h>
       #include <sys/stat.h>
       #include <time.h>
       #include <stdio.h>
       #include <stdlib.h>
       #include <fcntl.h>
       #include <grp.h>
       #include<pwd.h>
    #include <string.h>
       int main(int argc, char *argv[])
       {
           struct stat sb;
           char aa[]="-format=%a", AA[]="-format=%A", bb[]="-format=%b", gg[]="-format=%g",ii[]="-format=%i";
           
           if (argc != 2&& argc!=4) {
               fprintf(stderr, "Usage: %s <pathname>\n", argv[0]);
               exit(EXIT_FAILURE);
           }
           
           
           if(argc == 4){
               
               if (stat(argv[3], &sb) == -1 && strcmp(argv[1],"-c")) {
                   perror("stat");
                   exit(EXIT_FAILURE);
               }

               if(strcmp(argv[2],aa)==0){
                   printf("-format=");
                   printf( (S_ISDIR(sb.st_mode)) ? "1" : "0");
                   printf("%o\n",(sb.st_mode)&(0xFFF) );
                   
               }
               else if(strcmp(argv[2],AA)==0){
                   printf("-format=");
                   printf( (S_ISDIR(sb.st_mode)) ? "d" : "-");
                   printf( (sb.st_mode & S_IRUSR) ? "r" : "-");
                   printf( (sb.st_mode & S_IWUSR) ? "w" : "-");
                   printf( (sb.st_mode & S_IXUSR) ? "x" : "-");
                   printf( (sb.st_mode & S_IRGRP) ? "r" : "-");
                   printf( (sb.st_mode & S_IWGRP) ? "w" : "-");
                   printf( (sb.st_mode & S_IXGRP) ? "x" : "-");
                   printf( (sb.st_mode & S_IROTH) ? "r" : "-");
                   printf( (sb.st_mode & S_IWOTH) ? "w" : "-");
                   printf( (sb.st_mode & S_IXOTH) ? "x" : "-");
                   printf("\n");
                   
                   
                   
               }
               else if(strcmp(argv[2],bb)==0){
                   printf("-format=");
                   printf("%lld",(long long)sb.st_blocks);
                   
                   
               }

               else if(strcmp(argv[2],gg)==0){
                   printf("-format=");
                   printf("%d\n",sb.st_gid);
                   
                   
               }
               else if(strcmp(argv[2],ii)==0){
                   printf("-format=");
                   printf("%ld\n",(long) sb.st_ino);
                   
                   
               }
               return 0;
               
               
           }
           
           
           
           
           if(argc == 2){
           if (stat(argv[1], &sb) == -1) {
               perror("stat");
               exit(EXIT_FAILURE);
           }
           printf("\"File: \"%s\n",argv[1]);
           printf("Size: %lld\t",(long long)sb.st_size);
           printf("\tBlocks: %lld\t",(long long)sb.st_blocks);
           printf("I/O Block:%ld\t",(long) sb.st_blksize);
           
           switch (sb.st_mode & S_IFMT) {
           case S_IFBLK:  printf("  Block Device\n");            break;
           case S_IFCHR:  printf("  Character Device\n");        break;
           case S_IFDIR:  printf("  Directory\n");               break;
           case S_IFIFO:  printf("  FIFO/pipe\n");               break;
           case S_IFLNK:  printf("  Symlink\n");                 break;
           case S_IFREG:  printf("  Regular File\n");            break;
           case S_IFSOCK: printf("  Socket\n");                  break;
           default:       printf("  Unknown?\n");                break;
           }
           //printf("Device: %u/%u",major(sb.st_dev),minor(sb.st_dev));
           printf("Device: %lxh/%ldd\t",sb.st_dev, sb.st_dev);
           printf("      Inode: %ld", (long) sb.st_ino);
           printf("      Link: %ld\n", (long) sb.st_nlink);
           
           
           printf("Access:(");
           printf( (S_ISDIR(sb.st_mode)) ? "1" : "0");
           printf("%o/",(sb.st_mode)&(0xFFF) );
           printf( (S_ISDIR(sb.st_mode)) ? "d" : "-");
           //printf( (sb.st_mode & S_IFMT) ? "d" : "-");
           printf( (sb.st_mode & S_IRUSR) ? "r" : "-");
           printf( (sb.st_mode & S_IWUSR) ? "w" : "-");
           printf( (sb.st_mode & S_IXUSR) ? "x" : "-");
           printf( (sb.st_mode & S_IRGRP) ? "r" : "-");
           printf( (sb.st_mode & S_IWGRP) ? "w" : "-");
           printf( (sb.st_mode & S_IXGRP) ? "x" : "-");
           printf( (sb.st_mode & S_IROTH) ? "r" : "-");
           printf( (sb.st_mode & S_IWOTH) ? "w" : "-");
           printf( (sb.st_mode & S_IXOTH) ? "x" : "-");
           printf(")");
           
           struct passwd  *pwd;
           struct group   *grp;
           
           if ((pwd = getpwuid(sb.st_uid)) != NULL)
               printf("  Uid: (%d/ %s)",sb.st_uid, pwd->pw_name);
           else
               printf("  Uid: (%d)", sb.st_uid);
           
           
           /* Print out group name if it is found using getgrgid(). */
           if ((grp = getgrgid(sb.st_gid)) != NULL)
               printf("   Gid: (%d/ %s)", sb.st_gid,grp->gr_name);
           else
               printf("   Gid: (%d)", sb.st_gid);
           
           printf("\n");



           printf("Access: %s", ctime(&sb.st_ctime));
           printf("Modify: %s", ctime(&sb.st_atime));
           printf("Change: %s", ctime(&sb.st_mtime));

           return 0;
           }
           return 0;
       }

