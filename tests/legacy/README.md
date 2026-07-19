# Legacy Test Holding Area

`tests/legacy` is excluded from the default pytest collection through `pytest.ini`.

These files are retained as compatibility evidence only. Each file is inventoried
in `artifacts/repository_cleanup/test_inventory.json` with category `LEGACY` and
cleanup action `ARCHIVE` until the behaviour is either migrated into an active
regression test or deleted with a replacement recorded in
`artifacts/repository_cleanup/test_deletion_manifest.json`.
