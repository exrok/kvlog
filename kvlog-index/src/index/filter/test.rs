use kvlog::encoding::LogFields;

use crate::index::test::{test_index, TestIndexWriter};
use crate::log;
use kvlog::{encoding::FieldBuffer, Encode, LogLevel, SpanInfo};
#[test]
fn forward_query_continuations() {
    let mut index = test_index();
    let reader = index.reader().clone();
    let mut forward_query = reader.forward_query(&[]);
    assert!(forward_query.next().is_none());

    let mut writer = TestIndexWriter::new(&mut index);
    crate::log!(writer; msg="Hello");

    assert!(forward_query.next().is_some());
    assert!(forward_query.next().is_none());
    crate::log!(writer; msg="ABC");
    assert_eq!(forward_query.next().unwrap().into_iter().next().unwrap().message(), b"ABC");
    crate::log!(writer; msg="Nice");

    assert!(forward_query.next().is_some());
    assert!(forward_query.next().is_none());
}
