<?php
header('Content-Type: application/json; charset=utf-8');

try {
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
        throw new Exception('Phương thức không hợp lệ');
    }

    if (!isset($_POST['type']) || !isset($_FILES['file'])) {
        throw new Exception('Thiếu dữ liệu tải lên');
    }

    $type = $_POST['type'];
    $file = $_FILES['file'];

    if ($file['error'] !== UPLOAD_ERR_OK) {
        throw new Exception('Lỗi tải tệp: ' . $file['error']);
    }

    $rootUpload = __DIR__ . '/uploads';
    if (!is_dir($rootUpload)) {
        mkdir($rootUpload, 0755, true);
    }

    $response = [
        'success' => true,
        'payload' => null
    ];

    if ($type === 'image') {
        $response['payload'] = handleImageUpload($file);
    } elseif ($type === 'audio') {
        $response['payload'] = handleAudioUpload($file);
    } else {
        throw new Exception('Loại tệp không được hỗ trợ');
    }

    echo json_encode($response);
} catch (Exception $e) {
    http_response_code(400);
    echo json_encode([
        'success' => false,
        'message' => $e->getMessage()
    ]);
}

function handleImageUpload($file)
{
    $allowed = ['jpg', 'jpeg', 'png', 'gif', 'webp'];
    $ext = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
    if (!in_array($ext, $allowed)) {
        throw new Exception('Định dạng ảnh không được hỗ trợ');
    }

    $uploadDir = __DIR__ . '/uploads/images';
    if (!is_dir($uploadDir)) {
        mkdir($uploadDir, 0755, true);
    }

    $filename = uniqid('img_', true) . '.' . $ext;
    $targetPath = $uploadDir . '/' . $filename;

    if (!move_uploaded_file($file['tmp_name'], $targetPath)) {
        throw new Exception('Không thể lưu ảnh');
    }

    return [
        'type' => 'image',
        'url' => '/uploads/images/' . $filename
    ];
}

function handleAudioUpload($file)
{
    $uploadDir = __DIR__ . '/uploads/audio';
    if (!is_dir($uploadDir)) {
        mkdir($uploadDir, 0755, true);
    }

    $tmpPath = $uploadDir . '/' . uniqid('raw_', true) . '.webm';
    if (!move_uploaded_file($file['tmp_name'], $tmpPath)) {
        throw new Exception('Không thể lưu âm thanh tạm thời');
    }

    $outputFilename = uniqid('audio_', true) . '.mp3';
    $outputPath = $uploadDir . '/' . $outputFilename;

    $ffmpeg = 'ffmpeg';
    $command = $ffmpeg .
        ' -y -i ' . escapeshellarg($tmpPath) .
        ' -vn -ar 44100 -ac 2 -b:a 128k ' . escapeshellarg($outputPath) .
        ' 2>&1';
    exec($command, $cmdOutput, $returnCode);
    @unlink($tmpPath);

    if ($returnCode !== 0 || !file_exists($outputPath)) {
        throw new Exception('Không thể chuyển đổi sang MP3. Kiểm tra cài đặt ffmpeg.');
    }

    $duration = null;
    $ffprobe = 'ffprobe';
    $probeCmd = $ffprobe .
        ' -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ' .
        escapeshellarg($outputPath) . ' 2>&1';
    $durationOutput = shell_exec($probeCmd);
    if ($durationOutput !== null) {
        $duration = (float)trim($durationOutput);
    }

    return [
        'type' => 'audio',
        'url' => '/uploads/audio/' . $outputFilename,
        'duration' => $duration
    ];
}

