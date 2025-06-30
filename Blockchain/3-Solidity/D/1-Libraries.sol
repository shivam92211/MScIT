// SPDX-License-Identifier: MIT
pragma solidity >=0.7.0 <0.9.0;

library MyMathLib {
    function sum(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b;
    }

    function exponent(uint256 a, uint256 b) public pure returns (uint256) {
        return a ** b;
    }
}



//  This commented code in another file shows how to use the library


// SPDX-License-Identifier: MIT
// pragma solidity >=0.7.0 <0.9.0;

// import "./MyMathLib.sol";

// contract UseLib {
//     function getSum(uint256 x, uint256 y) public pure returns (uint256) {
//         return MyMathLib.sum(x, y);
//     }

//     function getExponent(uint256 x, uint256 y) public pure returns (uint256) {
//         return MyMathLib.exponent(x, y);
//     }
// }
