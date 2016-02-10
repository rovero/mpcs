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

$query = "SELECT roomId FROM (SELECT roomId FROM (SELECT r.id AS roomId, o.id AS oprId, o.date_time AS time FROM Operating_Room r JOIN Operation o WHERE r.operation_id = o.id) info WHERE time != '$date $time') available ORDER BY RAND() LIMIT 1";
$result = mysqli_query($dbcon, $query)
  or die('Query failed: ' . mysqli_error($dbcon));
$tuple = mysqli_fetch_array($result, MYSQL_ASSOC);
if($result->num_rows==0){
   echo "No room available at this time";
}
else{
  $insertOperation = "CALL addOperation($pId, '$date $time')";
  $operationResult = mysqli_query($dbcon, $insertOperation)
    or die('Query failed: ' . mysqli_error($dbcon));
    $rId = $tuple['roomId'];
  $insertRoom = "CALL addRoom($rId,$pId,'$date $time')";
  $roomResult = mysqli_query($dbcon, $insertRoom)
      or die('Query failed: ' . mysqli_error($dbcon));
  echo "Record updated successfully"."<br>";
  print "You have reserved room ".$tuple['roomId'];
}
// Free result
mysqli_free_result($result);

// Closing connection
mysqli_close($dbcon);
?>
