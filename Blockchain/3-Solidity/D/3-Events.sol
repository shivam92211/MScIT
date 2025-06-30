// SPDX-License-Identifier: MIT
pragma solidity ^0.5.0;

// Creating a contract
contract EventExample {
    // Declaring state variable
    uint256 public value = 0;

    // Declaring an event
    event Increment(address owner);

    // Function to emit event and update value
    function getValue(uint256 _a, uint256 _b) public {
        emit Increment(msg.sender); // Log caller's address
        value = _a + _b;             // Update state
    }
}
