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

$exp = $_REQUEST['exp'];
$query = "SELECT d.name AS doctor_name FROM Doctor d JOIN ParticipatesIn p ON d.work_id = p.doctor_work_id WHERE department_id = $exp GROUP BY doctor_name ORDER BY COUNT(*) DESC LIMIT 10;";
$result = mysqli_query($dbcon, $query)
  or die('Query failed: ' . mysqli_error($dbcon));

while($row = $result->fetch_array()){
    echo $row['doctor_name'];
      echo "<br />";
}
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>
