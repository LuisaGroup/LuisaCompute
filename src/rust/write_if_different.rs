fn canonicalize_line_ending(content: &str) -> String {
    let mut result = String::new();
    for line in content.lines() {
        result.push_str(line);
        result.push('\n');
    }
    result
}

pub fn write_if_different(path: &String, content: Vec<u8>) {
    // convert content to string
    let content = match String::from_utf8(content) {
        Ok(c) => c,
        Err(_) => return,
    };
    let cur_content = match std::fs::read(path) {
        Ok(c) => c,
        Err(_) => Vec::new(),
    };
    let cur_content = match String::from_utf8(cur_content) {
        Ok(c) => c,
        Err(_) => return,
    };
    let content = canonicalize_line_ending(&content);
    let cur_content = canonicalize_line_ending(&cur_content);
    if content != cur_content {
        match std::fs::write(path, content) {
            Ok(_) => {}
            Err(_) => {}
        }
    }
}
