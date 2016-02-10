<?php

// Connection parameters
$host = 'cspp53001.cs.uchicago.edu';
$username = 'qydeng';
$password = 'TEST';
$database = $username.'DB';

// Attempting to connect
$dbcon = mysqli_connect($host, $username, $password, $database)
   or die('Could not connect: ' . mysqli_connect_error());
print 'Connected successfully!<br>';

// Getting the input parameter (user):

$pId = $_REQUEST['pId'];
$date = $_REQUEST['date'];
$time = $_REQUEST['time'];

$query = "SELECT id FROM Operation WHERE patient_id = $pId AND date_time = '$date $time'";

if ($findId=mysqli_query($dbcon,$query))
    {
    // Return the number of rows in result set
    $rowcount=mysqli_num_rows($findId);
      if($rowcount==0){
        echo "Reservation not found!";
      }
      else{
        $sql = "CALL delReservation($pId, '$date $time')";
        if ($dbcon->query($sql) === TRUE) {
          echo "Reservation cancelled successfully";
        }
        else {
          echo "Error deleting record: " . $dbcon->error;
        }
      }
    }


// Free result
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>
