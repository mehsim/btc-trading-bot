import inspect
import main


def test_no_unbound_local_get_orderbook_imbalance_in_main():
    """Verify that get_orderbook_imbalance is not shadowed or treated as an unbound local inside main()."""
    # Inspect local variable names of main()
    main_code = main.main.__code__
    local_varnames = main_code.co_varnames

    # get_orderbook_imbalance should NOT be in main's local variable table (co_varnames)
    assert "get_orderbook_imbalance" not in local_varnames, (
        "get_orderbook_imbalance is in main's local variables table (co_varnames), "
        "which causes UnboundLocalError when accessed before local assignment."
    )

    # Verify get_orderbook_imbalance is accessible in main's global scope
    assert callable(getattr(main, "get_orderbook_imbalance", None))


def test_bybit_get_orderbook_imbalance_imported():
    """Verify bybit_get_orderbook_imbalance is callable and exported in main."""
    assert callable(getattr(main, "bybit_get_orderbook_imbalance", None))
