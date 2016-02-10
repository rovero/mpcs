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

$deptpopular = $_REQUEST['deptpopular'];
$query = "SELECT d.name AS doctor_name FROM Cures c JOIN Doctor d WHERE department_id = $deptpopular GROUP BY name ORDER BY COUNT(*) DESC LIMIT 10";
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
