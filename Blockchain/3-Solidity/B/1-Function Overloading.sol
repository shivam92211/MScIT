// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OverloadingExample {
    // Adds two numbers
    function add(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b;
    }

    // Concatenates two strings
    function add(string memory a, string memory b) public pure returns (string memory) {
        return string(abi.encodePacked(a, b));
    }
}
