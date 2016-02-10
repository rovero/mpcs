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

$pd = $_REQUEST['pd'];
$query = "SELECT first_name, last_name FROM Cures c JOIN Patient p ON c.patient_id = p.id WHERE p.doctor_work_id = $pd";
$result = mysqli_query($dbcon, $query)
  or die('Query failed: ' . mysqli_error($dbcon));


while($row = $result->fetch_array()){
    echo $row["first_name"]." ".$row["last_name"];
      echo "<br />";
}
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>
