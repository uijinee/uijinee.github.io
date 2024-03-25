---
title: "3. File System(disk)"
date: 2023-11-03 22:00:00 +0900
categories: ["Computer Science", "Operating System"]
tags: ["os", "operating system"]
use_math: true
---

Computer가 주로 사용하는 Memory에는 2종류가 있다.
- Primary Memory<br>
    *ex. RAM, ROM, ...*
- Secondary Memory<br>
    *ex. HDD, SSD, CD, ...*

File System은 여기서 Secondary Memory를 효율적으로 사용하기 위한 방법으로 다음 2가지에 초점을 두고 있다
- **Space Efficiency**
- **Time Efficiency**

이 File System에는 다음과 같이 여러 종류가 있지만, 여기서는 "linux2.4"의 Default File System인 **"EXT2"** 에 대해서 알아보자.

- File System의 종류
    - EXT2, EXT3, EXT4
    - FAT
    - NTFS

---
# On-Disk File System(EXT2)

## 1. Concept

### 1) Concept

![Alt text](/assets/img/post/operating_system/fs_ext2_concept_detail.png)

> 앞서 정의한 대로 File System에서는 file을 어떻게 Secondary Memory에 효율적으로 저장할 것인가가 중요하다.
>
> EXT2에서 채택한 Efficiency는 다음과 같다.
>   - Space Efficiency<br>
>   : File을 Block으로 나누고 이를 Disk의 빈 공간에 Scatter하여 저장한다.
>   - Time Efficiency<br>
>   : Inode Table을 통해 접근해야 할 Block의 위치를 정리해 놓는다.
>
> ---
> #### Delete
>
> ![Alt text](/assets/img/post/operating_system/fs_delete.png)
>
> 만약 어떤 File을 Directory에서 delete하고자 한다면 다음의 3가지를 바꿈으로써 간단하게 구현이 가능하다.
>
> - DBM
> - IBM
> - Directory(File Inode)<br>
>   : Directory속 File의 실제 위치를 가리키는 Inode를 0으로 바꾼다.<br>
>   (0은 지워졌다는 의미로써 사용된다.)
> 
>
>> 즉, Disk에서 해당 File의 내용을 실제로 지우는 것이 아니라,<br>
>> directory와 File의 Inode연결만 끊어줌으로써, Time Efficiency를 효과적으로 구현할 수 있다.
> ---

### 2) 파일 종류

![Alt text](/assets/img/post/operating_system/file_type.png)

> EXT2에서는 프로그램 뿐만 아니라 Socket이나 기타 Device등 모든 객체를 File로써 관리한다.
>
> 이때, File에는 다음과 같이 4종류의 File이 존재한다.
> 
> ---
> - Regular File<br>
>   : 일반적인 File들 *(ex. text, program, graphic)*
> - Directory File<br>
>   : Member File들의 정보를 가지고 있는 File들 *(ex. directory)*
> - SymbolicLink File
> - Special File<br>
>   : 특수 File들 *(ex. Device, Socket, Pipe)*

---
## 2. File System의 구성

![Alt text](/assets/img/post/operating_system/fs_ext2_concept_detail.png)

*(참고: Inode Block의 번호는 0번부터시작하지만 Data Block의 번호는 1번부터 시작한다.)*

### 1) Super Block(1 blk)

```cpp
/* include/linux/ext2_fs.h */
struct ext2_super_block {
    __le32 s_inodes_count;      /* Inodes count */
    __le32 s_blocks_count;      /* Blocks count */

    __le32 s_free_blocks_count; /* Free blocks count */
    __le32 s_free_inodes_count; /* Free inodes count */
    
    __le32 s_first_data_block;  /* First Data Block */
    __le32 s_log_block_size;    /* Block size */

    __le16 s_magic;             /* Magic signature */
    ...
}
```
> #### Define
>  
> Super Block은 File System의 Global Information을 갖고 있는 Block이다.
> 
>> Super Block은 1 Block을 차지하고 첫번째 Block에 위치한다.
>
> ---
> #### Info
> ![Alt text](/assets/img/post/operating_system/superblock.png)
> 
> - *(little endian, 32bit)* **s_inodes_count** <br>
>   : Disk안의 Total Inode의 개수
> - *(little endian, 32bit)* **s_blocks_count** <br>
>   : Disk안의 Total Block의 개수를 의미한다.
> - *(little endian, 32bit)* **s_first_data_block** <br>
>   : SuperBlock이 위치하는 Block #
> - *(little endian, 32bit)* **s_log_block_size** <br>
>   : Block 한개 당 크기 ($1024*2^{s\_log\_block\_size}$)
> - *(little endian, 16bit)* **s_magic** <br>
>   : 사용하는 파일 시스템의 형식(ext2, ext3, ...)

### 2) Group Descriptor(n blk)

```cpp
/* include/linux/ext2_fs.h */
struct ext2_group_desc {
    __le32 bg_block_bitmap;      /* Blocks bitmap block */
    __le32 bg_inode_bitmap;      /* Inodes bitmap block */
    __le32 bg_inode_table;       /* Inodes table block */
    ...
};
```

> #### Define
> 
> Data Block Bitmap과 Inode Bitmap, Inode Table의 위치를 저장하고 있는 Block이다.
> 
>> Group Descriptor는 N Block을 차지하고 SuperBlock의 바로 다음에 위치한다.
>
> ---
> #### Info
> ![Alt text](/assets/img/post/operating_system/groupdescriptor.png)
>
> - *(little endian, 32bit)* **bg_block_bitmap** <br>
>   : Data Block Bitmap의 Block #
> - *(little endian, 32bit)* **bg_inode_bitmap** <br>
>   : Inode Bitmap의 Block #
> - *(little endian, 32bit)* **bg_inode_table** <br>
>   : Inode Table의 시작 Block #
>
> DBM과 IBM은 모두 1 Block씩만 할당되지만,<br>
> Inode Table은 N Block으로 할당된다.
> 
> ---
> #### Example
>
> *bg_block_bitmap=8<br>
> &rarr; $8*block\_size=8192(0x 2000)$<br>
> &rarr; 즉, $0x 2000$부터 1 Block이 DBM*

### 3) Data Block Bitmap(1 blk)


> #### Define
> 
> Data Block이 사용되고 있는지에 대한 정보를 Bitmap으로 적어둔 Block을 의미한다.
>
>> 1 Block을 차지하고 Block 전체를 Little Endian으로 해석하면 된다.
>
> ---
> #### Example
>
> ![Alt text](/assets/img/post/operating_system/dbm.png)
> 
> 위 경우와 같을 때,
>
> $0x 7f /ff /ff /ff /ff /ff$이므로 <br>
> $01111111/ 11111111/ 11111111/ 11111111/ 11111111/ 11111111$<br>
> 즉, 1번 ~ 47번 Block은 사용중이고 48번 Block이 비어 있음을 알 수 있다.

### 4) Inode Bitmap(1 blk)

> #### Define
> 
> Inode가 사용되고 있는지에 대한 정보를 Bitmap으로 적어둔 Block을 의미한다.
>
>> 1 Block을 차지하고 Block 전체를 Little Endian으로 해석하면 된다.
>
> ---
> #### Example
>
> ![Alt text](/assets/img/post/operating_system/inode_bitmap.png)
> 
> 위 경우와 같을 때,
>
> $0x ...0000 / 00 / 0f /ff$이므로 <br>
> $...00001111 / 11111111$<br>
> 즉, 1번 ~ 12번 Block은 사용 중이고 13번 ~ 184번 Block이 비어 있음을 알 수 있다.
>
> *(위의 경우 s_inode_count가 184였기 때문에 나머지 Block들은 채워져 있다고 나타내고 있다.)*

### 5) Inode Table(n blk)
```cpp
/* include/linux/ext2_fs.h */
struct ext2_inode {
    __le16 i_mode;        /* File mode */
    __le16 i_uid;         /* Low 16 bits of Owner Uid */
    __le32 i_size;        /* Size in bytes */
    __le32 i_atime;       /* Access time */
    __le32 i_ctime;       /* Creation time */
    __le32 i_mtime;       /* Modification time */
    __le32 i_dtime;       /* Deletion Time */
    __le16 i_gid;         /* Low 16 bits of Group Id */
    __le16 i_links_count; /* Links count */
    __le32 i_blocks;      /* Blocks count */
    __le32 i_flags;       /* File flags */
    union {
        struct {
            __le32 l_i_reserved1;
        } linux1;
        struct {
            __le32 h_i_translator;
        } hurd1;
        struct {
            __le32 m_i_reserved1;
        } masix1;
    } osd1;                        /* OS dependent 1 */
    __le32 i_block[EXT2_N_BLOCKS]; /* Pointers to blocks */
    ...
};
```

> #### Define
> 
> 각 File들의 위치를 저장하고 있는 Table이다.
> 
>> Inode Table은 N Block을 차지하고, Inode 1개의 크기는 $0x80=128$ Byte이다.
>>
>> 또, 1번 Inode는 Super Block이 사용하도록 예약되어 있고, 2번 Inode는 Root Directory가 사용하도록 예약되어 있다.
>
> ---
> #### Info
> ![Alt text](/assets/img/post/operating_system/inodetable.png)
>
> - *(little endian, 16bit)* **i_uid** <br>
>   : 이 File의 Owner
> - *(little endian, 32bit)* **i_size** <br>
>   : Inode Bitmap의 Block #
> - *(little endian, 32bit)* **i_block[EXT2_N_BLOCKS]** <br>
>   : 이 File의 내용이 존재하는 Block #
>   *(EXT2에서 EXT2_N_BLOCKS의 기본값은 15이다)*
>
> DBM과 IBM은 모두 1 Block씩만 할당되지만,<br>
> Inode Table은 N Block으로 할당된다.
> 
> ---
> #### Example
>
> i_block=\[$0x00000021$, $0x00000000$, ...\]<br>
> &rarr; 이 File의 Data는 $0x00000021(=33)$번째 Block에만 존재한다.


---
## 3. File구성

### 1) direcoty file

```cpp
struct ext2_dir_entry_2{
    __le32 inode;
    __le16 rec_len;
    __u8 name_len;
    __u8 file_type;
    char name[EXT2_NAME_LEN];
}
```

> #### Define
>
> linux의 directory File은 위와 같은 구조로 설계되어 있다.
>
>> Inode도 함께 저장되어 있는 것을 확인할 수 있다.
>
> ---
> #### Example
>
> ![Alt text](/assets/img/post/operating_system/linux_directory_example.png)

---
# linux command

File System을 공부하기 위해 알아야 할 linux명령어들은 다음과 같다.

### 1) xxd
```bash
xxd [옵션] [input파일]
```
1. 기능<br>
    : input 파일의 내용을 16진수로 바꾸어 보여준다.
2. Argument<br>
    `input파일`: 16진수로 해석할 File경로
2. Option<br>
    `-g1`: 1 byte단위로 해석하여 출력함

### 2) dd

```bash
dd bs=[Block size] count=[Block 개수] if=[Input File] of=[Output File]
```
1. 기능<br>
    : Block단위로 File을 복사하거나 File변환을 수행
2. Argument<br>
    `bs`: 한번에 읽고 쓸 최대 Byte의 크기(Block Size)<br>
    `count`: 복사할 Block의 개수<br>
    `if`: 입력 파일 경로<br>
    `of`: 출력 파일 경로

### 3) mkfs
```bash
mkfs -t [FileSystem의 종류] [FileSystem을 생성할 Partition]
```

1. 기능<br>
    : 지정된 File System Type을 활용해 Partition에 가상의 FileSystem을 생성.
2. Argument<br>
    `FileSystem을 생성할 Partition`: File System이 동작하기 위한 파티션
2. Option<br>
    `-t`: 생성할 FileSystem의 종류<br>
    (`btrfs`: B-tree File System, `ntfs`: NTFS(주로 Window에서 사용), `ext2`: 두번째 확장파일 시스템)

### 4) mount

```bash
mount [옵션] [device(disk)이름] [mount할 directory]
```

1. 기능<br>
    : 특정 device를 directory에 연결하는 명령어<br>
    해당 directory를 통해 이와 연결된 device에 접근할 수 있다.
2. Argument<br>
    `FileSystem을 생성할 Partition`: File System이 동작하기 위한 파티션
2. Option<br>
    `-o loop`: loop device(, virtual disk)를 mount하는 경우 사용

> #### mount
>
> - Windows<br>
>   : 물리 Device(USB, 외장하드)를 컴퓨터와 연결하면 이 Device를 자동으로 인식함.
>
> - Linux<br>
>   : linux는 File뿐만 아니라 Device도 하나의 File로써 처리한다.<br>
>   따라서 Device를 사용하기 위해서는 이를 특정 File에 연결해서 사용해야 하는데, 이 연결 과정을 mount라고 한다.
>
> ---
> #### loop device
>
> linux에서 file을 Block Device로써 접근이 가능하도록 만드는 가상의 Device를 의미한다.<br>
> 즉, 쉽게말해 File을 하드 디스크처럼 다루게해주는 기능을 가진 가상의 Device를 말한다.
>
> ---

### 5) umount

```bash
umount [directory 이름]
```
1. 기능<br>
    : directory에 연결된 device의 mount를 해제함
2. Argument<br>
    `directory 이름`: mount해제를 원하는 device가 연결된 directory

> #### umount의 이유
> 
> device의 mount를 해제하지 않고 device를 제거하거나 device의 내용을 확인하면 하면 다음과 같은 문제점이 발생한다.
> 
> - **Data입력**<br>
>    : umount는 mount되어 있는 동안 Device의 변경된 내용을 Disk에 실제로 기록 완료하는 과정을 포함하기 때문에 umount하지 않을 경우 data 손상 발생 가능<br>
>   *(Device의 변경된 내용을 Disk에 실제로 기록하기 전에 Cache에만 저장하고 실제 반영은 후에 하는 경우가 존재함)*
> 
> - **lock**<br>
>    : Device가 Directory에 mount된 동안 Device는 잠기기 때문에 다른 작업에서 접근이 불가능함.