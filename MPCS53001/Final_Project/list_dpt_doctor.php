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

// Selecting a database
//   Unnecessary in this case because we have already selected
//   the right database with the connect statement.
mysqli_select_db($dbcon, $database)
   or die('Could not select database');
print 'Selected database successfully!<br>';
$dpt = $_REQUEST['dpt'];

$query = "SELECT work_id,d.name AS d_name, dept.name AS department FROM Doctor d JOIN Department dept ON dept.id = d.department_id WHERE d.department_id = $dpt";
$result = mysqli_query($dbcon, $query)
   or die('Query failed: ' . mysqli_error($dbcon));

// Can also check that there is only one tuple in the result

print "This department has the following doctors:<br/>";

while($row = $result->fetch_array()){
    echo $row['work_id'] . "|" . $row['d_name']. "|" . $row['department'];
      echo "<br />";
}
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>
