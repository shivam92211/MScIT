// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.16 <0.9.0;

contract InlineAssembly {
    // Function using inline assembly to add 16 to input
    function add(uint256 a) public pure returns (uint256 b) {
        assembly {
            b := add(a, 16)
        }
    }
}
